import io
import logging
import asyncio
import traceback
import html
import json
from datetime import datetime
import openai

import telegram
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    TypeHandler,
    ApplicationHandlerStop,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils

import base64

# setup
db = database.Database()
logger = logging.getLogger(__name__)

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """Commands:
⚪ /retry – Regenerate last bot answer
⚪ /new – Start new dialog
⚪ /chats – Show & switch past dialogs
⚪ /imagine &lt;prompt&gt; – Generate an image from any mode
⚪ /mode – Select chat mode
⚪ /settings – Show settings
⚪ /balance – Show balance
⚪ /help – Show help

🎨 Generate images from text prompts in <b>👩‍🎨 Artist</b> /mode
👥 Add bot to <b>group chat</b>: /help_group_chat
🎤 You can send <b>Voice Messages</b> instead of text
"""

HELP_GROUP_CHAT_MESSAGE = """You can add bot to any <b>group chat</b> to help and entertain its participants!

Instructions (see <b>video</b> below):
1. Add the bot to the group chat
2. Make it an <b>admin</b>, so that it can see messages (all other rights can be restricted)
3. You're awesome!

To get a reply from the bot in the chat – @ <b>tag</b> it or <b>reply</b> to its message.
For example: "{bot_username} write a poem about Telegram"
"""


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", config.models["available_text_models"][0])

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int) or isinstance(n_used_tokens, float):  # old format
        new_n_used_tokens = {
            "gpt-3.5-turbo": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)

    # image generation
    if db.get_user_attribute(user.id, "n_generated_images") is None:
        db.set_user_attribute(user.id, "n_generated_images", 0)


async def is_bot_mentioned(update: Update, context: CallbackContext):
     try:
         message = update.message

         if message.chat.type == "private":
             return True

         if message.text is not None and ("@" + context.bot.username) in message.text:
             return True

         if message.reply_to_message is not None:
             if message.reply_to_message.from_user.id == context.bot.id:
                 return True
     except Exception:
         return True
     else:
         return False


async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id

    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)

    reply_text = "Hi! I'm <b>ChatGPT</b> bot implemented with OpenAI API 🤖\n\n"
    reply_text += HELP_MESSAGE

    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)


async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def help_group_chat_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update, context, update.message.from_user)
     user_id = update.message.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)

     await update.message.reply_text(text, parse_mode=ParseMode.HTML)
     await update.message.reply_video(config.help_group_chat_video_path)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("No message to retry 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)

async def _vision_message_handle_fn(
    update: Update, context: CallbackContext, use_new_dialog_timeout: bool = True
):
    logger.info('_vision_message_handle_fn')
    user_id = update.message.from_user.id
    current_model = db.get_user_attribute(user_id, "current_model")

    if not config.models["info"][current_model].get("vision", False):
        await update.message.reply_text(
            "🥲 Image understanding is only available for <b>vision-capable</b> models (e.g. GPT-4o, GPT-4o mini, GPT-5.5 or Claude). Please change your model in /settings",
            parse_mode=ParseMode.HTML,
        )
        return

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    # new dialog timeout
    if use_new_dialog_timeout:
        if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
            db.start_new_dialog(user_id)
            await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    buf = None
    if update.message.effective_attachment:
        photo = update.message.effective_attachment[-1]
        photo_file = await context.bot.get_file(photo.file_id)

        # store file in memory, not on disk
        buf = io.BytesIO()
        await photo_file.download_to_memory(buf)
        buf.name = "image.jpg"  # file extension is required
        buf.seek(0)  # move cursor to the beginning of the buffer

    # in case of CancelledError
    n_input_tokens, n_output_tokens = 0, 0

    try:
        # send placeholder message to user
        placeholder_message = await update.message.reply_text("...")
        message = update.message.caption or update.message.text or ''

        # send typing action
        await update.message.chat.send_action(action="typing")

        dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
        parse_mode = {"html": ParseMode.HTML, "markdown": ParseMode.MARKDOWN}[
            config.chat_modes[chat_mode]["parse_mode"]
        ]

        chatgpt_instance = openai_utils.ChatGPT(model=current_model)
        if config.enable_message_streaming:
            gen = chatgpt_instance.send_vision_message_stream(
                message,
                dialog_messages=dialog_messages,
                image_buffer=buf,
                chat_mode=chat_mode,
            )
        else:
            (
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = await chatgpt_instance.send_vision_message(
                message,
                dialog_messages=dialog_messages,
                image_buffer=buf,
                chat_mode=chat_mode,
            )

            async def fake_gen():
                yield "finished", answer, (
                    n_input_tokens,
                    n_output_tokens,
                ), n_first_dialog_messages_removed

            gen = fake_gen()

        prev_answer = ""
        async for gen_item in gen:
            (
                status,
                answer,
                (n_input_tokens, n_output_tokens),
                n_first_dialog_messages_removed,
            ) = gen_item

            answer = answer[:4096]  # telegram message limit

            # update only when 100 new symbols are ready
            if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                continue

            try:
                await context.bot.edit_message_text(
                    answer,
                    chat_id=placeholder_message.chat_id,
                    message_id=placeholder_message.message_id,
                    parse_mode=parse_mode,
                )
            except telegram.error.BadRequest as e:
                if str(e).startswith("Message is not modified"):
                    continue
                else:
                    await context.bot.edit_message_text(
                        answer,
                        chat_id=placeholder_message.chat_id,
                        message_id=placeholder_message.message_id,
                    )

            await asyncio.sleep(0.01)  # wait a bit to avoid flooding

            prev_answer = answer

        # update user data
        if buf is not None:
            base_image = base64.b64encode(buf.getvalue()).decode("utf-8")
            new_dialog_message = {"user": [
                        {
                            "type": "text",
                            "text": message,
                        },
                        {
                            "type": "image",
                            "image": base_image,
                        }
                    ]
                , "bot": answer, "date": datetime.now()}
        else:
            new_dialog_message = {"user": [{"type": "text", "text": message}], "bot": answer, "date": datetime.now()}
        
        db.set_dialog_messages(
            user_id,
            db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
            dialog_id=None
        )

        db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

    except asyncio.CancelledError:
        # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
        db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
        raise

    except Exception as e:
        error_text = f"Something went wrong during completion. Reason: {e}"
        logger.error(error_text)
        await update.message.reply_text(error_text)
        return

async def unsupport_message_handle(update: Update, context: CallbackContext, message=None):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    error_text = f"I don't know how to read files or videos. Send the picture in normal mode (Quick Mode)."
    logger.error(error_text)
    await update.message.reply_text(error_text)
    return

async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id

    # admin flow: capturing a username/id to add to the allowlist
    if context.user_data.get("awaiting_user_add") and _is_admin(update.message.from_user):
        context.user_data["awaiting_user_add"] = False
        raw = (_message or "").strip()
        if not raw:
            await update.message.reply_text("Empty input, nothing added.")
            return
        entry = int(raw) if raw.lstrip("-").isdigit() else _norm_entry(raw)
        db.add_allowed(entry)
        disp = ("@" + entry) if isinstance(entry, str) else str(entry)
        await update.message.reply_text(f"✅ Added to allowlist: {disp}")
        return

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    if chat_mode == "artist":
        await generate_image_handle(update, context, message=message)
        return

    current_model = db.get_user_attribute(user_id, "current_model")

    async def message_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Starting new dialog due to timeout (<b>{config.chat_modes[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("...")

            # send typing action
            await update.message.chat.send_action(action="typing")

            if _message is None or len(_message) == 0:
                 await update.message.reply_text("🥲 You sent <b>empty message</b>. Please, try again!", parse_mode=ParseMode.HTML)
                 return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            if config.enable_message_streaming:
                gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            else:
                answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                    _message,
                    dialog_messages=dialog_messages,
                    chat_mode=chat_mode
                )

                async def fake_gen():
                    yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                gen = fake_gen()

            prev_answer = ""
            
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item

                answer = answer[:4096]  # telegram message limit
                    
                # update only when 100 new symbols are ready
                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue

                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("Message is not modified"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding
                
                prev_answer = answer
            
            # update user data
            new_dialog_message = {"user": [{"type": "text", "text": _message}], "bot": answer, "date": datetime.now()}

            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"Something went wrong during completion. Reason: {e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Your current dialog is too long, so your <b>first message</b> was removed from the context.\n Send /new command to start new dialog"
            else:
                text = f"✍️ <i>Note:</i> Your current dialog is too long, so <b>{n_first_dialog_messages_removed} first messages</b> were removed from the context.\n Send /new command to start new dialog"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        model_supports_vision = config.models["info"][current_model].get("vision", False)
        photo_sent = update.message.photo is not None and len(update.message.photo) > 0
        # only take the vision path when a photo was actually sent — otherwise text
        # (incl. transcribed voice) wrongly hit effective_attachment[-1] and crashed
        if photo_sent:
            if not model_supports_vision:
                # a photo was sent but the selected model can't read images:
                # fall back to a vision-capable default
                current_model = "gpt-4o"
                db.set_user_attribute(user_id, "current_model", "gpt-4o")
            task = asyncio.create_task(
                _vision_message_handle_fn(update, context, use_new_dialog_timeout=use_new_dialog_timeout)
            )
        else:
            task = asyncio.create_task(
                message_handle_fn()
            )

        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ Canceled", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]


async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ Please <b>wait</b> for a reply to the previous message\n"
        text += "Or you can /cancel it"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False


async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)
    
    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(update, context, message=transcribed_text)


async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    await update.message.chat.send_action(action="upload_photo")

    message = message or update.message.text

    try:
        images = await openai_utils.generate_images(message, n_images=config.return_n_generated_images, size=config.image_size)
    except openai.BadRequestError as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 Your request <b>doesn't comply</b> with OpenAI's usage policies.\nWhat did you write there, huh?"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise

    # usage accounting
    db.set_user_attribute(user_id, "n_generated_images", len(images) + db.get_user_attribute(user_id, "n_generated_images"))

    for image in images:
        await update.message.chat.send_action(action="upload_photo")
        await update.message.reply_photo(io.BytesIO(image))


async def imagine_handle(update: Update, context: CallbackContext):
    # generate an image from any chat mode (unlike the artist-mode-only path)
    prompt = " ".join(context.args).strip() if context.args else ""
    if not prompt:
        await update.message.reply_text(
            "Использование: <code>/imagine текст запроса</code>\n"
            "Например: <code>/imagine рыжий кот на Таймс-сквер, иллюстрация</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    await generate_image_handle(update, context, message=prompt)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.set_user_attribute(user_id, "current_model", "gpt-4o-mini")

    db.start_new_dialog(user_id)
    await update.message.reply_text("Starting new dialog ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    context.user_data["awaiting_user_add"] = False

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await update.message.reply_text("<i>Nothing to cancel...</i>", parse_mode=ParseMode.HTML)


def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"Select <b>chat mode</b> ({len(config.chat_modes)} modes available):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_chat_mode_menu(0)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
     if await is_previous_message_not_answered_yet(update.callback_query, context): return

     user_id = update.callback_query.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     query = update.callback_query
     await query.answer()

     page_index = int(query.data.split("|")[1])
     if page_index < 0:
         return

     text, reply_markup = get_chat_mode_menu(page_index)
     try:
         await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
     except telegram.error.BadRequest as e:
         if str(e).startswith("Message is not modified"):
             pass


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML
    )


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models (chunked into rows of 2 to stay within
    # Telegram's per-row button limit as the model list grows)
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )

    keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]

    # dialog-switch history selector. Telegram can't put text between keyboard
    # rows, so a header-row button labels the option group below it.
    current_n = _get_show_last_n(user_id)
    keyboard.append([InlineKeyboardButton("— 📜 Reprint history on /chats switch —", callback_data="settings_noop")])
    n_buttons = []
    for n in LAST_N_OPTIONS:
        title = "Off" if n == 0 else str(n)
        if n == current_n:
            title = "✅ " + title
        n_buttons.append(InlineKeyboardButton(title, callback_data=f"set_lastn|{n}"))
    keyboard.append(n_buttons)

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def set_show_last_n_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    n = int(query.data.split("|")[1])
    if n not in LAST_N_OPTIONS:
        return
    db.set_user_attribute(user_id, "show_last_n_on_switch", n)

    text, reply_markup = get_settings_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def settings_noop_handle(update: Update, context: CallbackContext):
    # header/label button in menus — does nothing but must answer the callback
    await update.callback_query.answer()


# ---------- access control (runtime allowlist) ----------

def _norm_entry(entry):
    return entry.lstrip("@").lower() if isinstance(entry, str) else entry


def _config_allow_entries():
    return [_norm_entry(x) for x in config.allowed_telegram_usernames]


def _is_admin(user) -> bool:
    # admins = entries from config (env-seeded owners); they manage the allowlist
    if user is None:
        return False
    entries = _config_allow_entries()
    if user.id in entries:
        return True
    if user.username and _norm_entry(user.username) in entries:
        return True
    return False


def _is_allowed(user) -> bool:
    if user is None:
        return False
    if _is_admin(user):
        return True
    entries = [_norm_entry(x) for x in db.get_extra_allowed()]
    if user.id in entries:
        return True
    if user.username and _norm_entry(user.username) in entries:
        return True
    return False


def _is_bot_invocation(update: Update, context: CallbackContext) -> bool:
    # is this update actually directed at the bot?
    if update.callback_query is not None:
        return True  # tapping an inline button on the bot's own message

    msg = update.effective_message
    chat = update.effective_chat
    if msg is None or chat is None:
        return False

    if chat.type == "private":
        return True  # every private message is addressed to the bot

    # group / supergroup: only @mention, reply-to-bot, or a command for this bot
    text = msg.text or msg.caption or ""
    bot_username = context.bot.username

    entities = list(msg.entities or []) + list(msg.caption_entities or [])
    for ent in entities:
        if ent.type == "bot_command" and ent.offset == 0:
            cmd = text[ent.offset: ent.offset + ent.length]
            if "@" not in cmd or cmd.endswith("@" + (bot_username or "")):
                return True

    if bot_username and ("@" + bot_username) in text:
        return True

    reply = msg.reply_to_message
    if reply is not None and reply.from_user is not None and reply.from_user.id == context.bot.id:
        return True

    return False


async def access_gate(update: Update, context: CallbackContext):
    # 1) in groups, ignore anything not addressed to the bot (no @tag / reply / command)
    if not _is_bot_invocation(update, context):
        raise ApplicationHandlerStop

    # empty allowlist everywhere => open to all (upstream behavior)
    if len(config.allowed_telegram_usernames) == 0 and len(db.get_extra_allowed()) == 0:
        return

    user = update.effective_user
    chat = update.effective_chat

    # 2) the bot was invoked — check access
    if user is not None and _is_allowed(user):
        return

    group_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int) and x < 0]
    if chat is not None and chat.id in group_ids:
        return

    # invoked but no rights — tell the user, then stop
    try:
        if update.callback_query is not None:
            await update.callback_query.answer("Нет доступа", show_alert=True)
        elif update.effective_message is not None:
            await update.effective_message.reply_text("⛔ У вас нет доступа к этому боту.")
    except Exception:
        pass
    raise ApplicationHandlerStop


def get_users_menu():
    text = "<b>Access control</b>\n\n"
    text += "Owners (from config, fixed):\n"
    for e in config.allowed_telegram_usernames:
        text += f"  🔒 {e}\n"

    extra = db.get_extra_allowed()
    text += "\nAdded users:\n"
    if not extra:
        text += "  (none)\n"

    keyboard = []
    for e in extra:
        disp = ("@" + e) if isinstance(e, str) else str(e)
        keyboard.append([InlineKeyboardButton(f"❌ {disp}", callback_data=f"rmuser|{e}")])
    keyboard.append([InlineKeyboardButton("➕ Add user", callback_data="adduser")])

    return text, InlineKeyboardMarkup(keyboard)


async def users_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if not _is_admin(update.message.from_user):
        await update.message.reply_text("⛔ Admins only.")
        return
    db.set_user_attribute(update.message.from_user.id, "last_interaction", datetime.now())
    text, reply_markup = get_users_menu()
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def add_user_handle(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    if not _is_admin(query.from_user):
        return
    context.user_data["awaiting_user_add"] = True
    await context.bot.send_message(
        query.message.chat.id,
        "Send <b>@username</b> or numeric <b>user id</b> to allow. /cancel to abort.",
        parse_mode=ParseMode.HTML,
    )


async def rm_user_handle(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    if not _is_admin(query.from_user):
        return
    raw = query.data.split("|", 1)[1]
    entry = int(raw) if raw.lstrip("-").isdigit() else raw
    db.remove_allowed(entry)
    text, reply_markup = get_users_menu()
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if not str(e).startswith("Message is not modified"):
            raise


DIALOGS_PER_PAGE = 4
DEFAULT_SHOW_LAST_N = 3
LAST_N_OPTIONS = [0, 3, 5, 10]


def _get_show_last_n(user_id: int) -> int:
    n = db.get_user_attribute(user_id, "show_last_n_on_switch")
    return DEFAULT_SHOW_LAST_N if n is None else int(n)


def _dialog_snippet(dialog: dict, limit: int = 50) -> str:
    messages = dialog.get("messages", [])
    snippet = ""
    if messages:
        try:
            snippet = messages[0]["user"][0]["text"]
        except (KeyError, IndexError, TypeError):
            snippet = ""
    snippet = (snippet or "…").replace("\n", " ").strip()
    if len(snippet) > limit:
        snippet = snippet[:limit] + "…"
    return snippet


def _dialog_meta(dialog: dict) -> str:
    start_time = dialog.get("start_time")
    dt_str = start_time.strftime("%d.%m %H:%M") if start_time else "?"
    model_key = dialog.get("model", "?")
    model_name = config.models["info"].get(model_key, {}).get("name", model_key)
    return f"{dt_str} · {model_name} · {len(dialog.get('messages', []))} msg"


def get_dialogs_menu(user_id: int, page_index: int):
    # skip empty dialogs (created by /new or model switches) — nothing to switch to
    dialogs = [d for d in db.get_dialogs(user_id, limit=200) if d.get("messages")]
    current_dialog_id = db.get_user_attribute(user_id, "current_dialog_id")

    if not dialogs:
        return "You have no dialogs with messages yet.", None

    text = f"Диалоги ({len(dialogs)}). Верхняя кнопка — открыть чат, нижняя (🗑) — удалить:"

    page_dialogs = dialogs[page_index * DIALOGS_PER_PAGE:(page_index + 1) * DIALOGS_PER_PAGE]

    keyboard = []
    for dialog in page_dialogs:
        # full-width row: leads with the first message so the chat is recognizable
        switch_label = "💬 " + _dialog_snippet(dialog)
        if dialog["_id"] == current_dialog_id:
            switch_label = "✅ " + switch_label
        keyboard.append([InlineKeyboardButton(switch_label, callback_data=f"set_dialog|{dialog['_id']}")])
        # full-width delete row carries the metadata (date · model · count)
        keyboard.append([InlineKeyboardButton("🗑 " + _dialog_meta(dialog), callback_data=f"deldlg|{dialog['_id']}")])

    # pagination
    if len(dialogs) > DIALOGS_PER_PAGE:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * DIALOGS_PER_PAGE >= len(dialogs))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_dialogs|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_dialogs|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_dialogs|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_dialogs|{page_index + 1}")
            ])

    return text, InlineKeyboardMarkup(keyboard)


async def show_dialogs_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_dialogs_menu(user_id, 0)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_dialogs_callback_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    if await is_previous_message_not_answered_yet(update.callback_query, context): return

    user_id = update.callback_query.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    query = update.callback_query
    await query.answer()

    page_index = int(query.data.split("|")[1])
    if page_index < 0:
        return

    text, reply_markup = get_dialogs_menu(user_id, page_index)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass


async def set_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    dialog_id = query.data.split("|", 1)[1]

    dialogs = {d["_id"]: d for d in db.get_dialogs(user_id, limit=1000)}
    dialog = dialogs.get(dialog_id)
    if dialog is None:
        await context.bot.send_message(query.message.chat.id, "Dialog not found.")
        return

    db.set_user_attribute(user_id, "current_dialog_id", dialog_id)

    start_time = dialog.get("start_time")
    dt_str = start_time.strftime("%d.%m %H:%M") if start_time else "?"
    model_key = dialog.get("model", "?")
    model_name = config.models["info"].get(model_key, {}).get("name", model_key)
    await context.bot.send_message(
        query.message.chat.id,
        f"🔀 <b>Switched to dialog</b> {dt_str} · {model_name}",
        parse_mode=ParseMode.HTML,
    )

    show_last_n = _get_show_last_n(user_id)
    if show_last_n <= 0:
        return

    recent = dialog.get("messages", [])[-show_last_n:]
    if not recent:
        await context.bot.send_message(query.message.chat.id, "(this dialog is empty)")
        return

    for message in recent:
        try:
            user_text = message["user"][0]["text"]
        except (KeyError, IndexError, TypeError):
            user_text = "…"
        bot_text = message.get("bot", "") or ""
        # plain text (no parse_mode) so special chars in history never break sending
        await context.bot.send_message(query.message.chat.id, f"👤 {user_text}"[:4000])
        if bot_text:
            await context.bot.send_message(query.message.chat.id, f"🤖 {bot_text}"[:4000])


async def del_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    dialog_id = query.data.split("|", 1)[1]

    deleted = db.delete_dialog(user_id, dialog_id)

    # if the active dialog was deleted, open a fresh one so state stays valid
    if db.get_user_attribute(user_id, "current_dialog_id") == dialog_id:
        db.start_new_dialog(user_id)

    await query.answer("Удалено" if deleted else "Не найдено")

    text, reply_markup = get_dialogs_menu(user_id, 0)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if not str(e).startswith("Message is not modified"):
            raise


async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_generated_images = db.get_user_attribute(user_id, "n_generated_images")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        # skip models that are no longer configured (e.g. removed legacy models
        # a user spent tokens on in the past) to avoid a KeyError
        if model_key not in config.models["info"]:
            continue

        n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key]["n_input_tokens"], n_used_tokens_dict[model_key]["n_output_tokens"]
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    # image generation
    image_generation_n_spent_dollars = config.models["info"]["gpt-image-1"]["price_per_1_image"] * n_generated_images
    if n_generated_images != 0:
        details_text += f"- GPT Image (image generation): <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

    total_n_spent_dollars += image_generation_n_spent_dollars

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"

    total_n_spent_dollars += voice_recognition_n_spent_dollars


    text = f"You spent <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"You used <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 Unfortunately, message <b>editing</b> is not supported"
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except Exception:
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "Start new dialog"),
        BotCommand("/chats", "Show & switch past dialogs"),
        BotCommand("/imagine", "Generate an image from a prompt"),
        BotCommand("/mode", "Select chat mode"),
        BotCommand("/retry", "Re-generate response for previous query"),
        BotCommand("/balance", "Show balance"),
        BotCommand("/settings", "Show settings"),
        BotCommand("/users", "Manage allowed users (admin)"),
        BotCommand("/help", "Show help message"),
    ])

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .post_init(post_init)
        .build()
    )

    # add handlers
    # access control is dynamic (runtime allowlist in DB) via a pre-gate,
    # so downstream handlers accept all updates and the gate drops disallowed ones
    user_filter = filters.ALL
    application.add_handler(TypeHandler(Update, access_gate), group=-1)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(MessageHandler(filters.VIDEO & ~filters.COMMAND & user_filter, unsupport_message_handle))
    application.add_handler(MessageHandler(filters.Document.ALL & ~filters.COMMAND & user_filter, unsupport_message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(CommandHandler("cancel", cancel_handle, filters=user_filter))

    application.add_handler(CommandHandler("chats", show_dialogs_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_dialogs_callback_handle, pattern="^show_dialogs"))
    application.add_handler(CallbackQueryHandler(set_dialog_handle, pattern="^set_dialog"))
    application.add_handler(CallbackQueryHandler(del_dialog_handle, pattern="^deldlg"))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))

    application.add_handler(CommandHandler("imagine", imagine_handle, filters=user_filter))

    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(show_chat_modes_callback_handle, pattern="^show_chat_modes"))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))
    application.add_handler(CallbackQueryHandler(set_show_last_n_handle, pattern="^set_lastn"))
    application.add_handler(CallbackQueryHandler(settings_noop_handle, pattern="^settings_noop$"))

    application.add_handler(CommandHandler("users", users_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(add_user_handle, pattern="^adduser$"))
    application.add_handler(CallbackQueryHandler(rm_user_handle, pattern="^rmuser"))

    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))

    application.add_error_handler(error_handle)

    # start the bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()
