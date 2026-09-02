import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, time, shutil, zipfile, random, logging, asyncio, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

pkgs = ["pyrogram", "tgcrypto", "huggingface_hub", "soundfile", "pyloudnorm", "librosa", "pydub", "numpy"]
for pkg in pkgs:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, create_repo
from pyrogram import Client, filters, idle
from pyrogram.types import Message, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8984996986:AAHT8KvQbQsiAlkIpBJayrcI5x6mxwDSDRU").strip()
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", "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").strip()
API_ID = int(os.getenv("TELEGRAM_API_ID", "38449618"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "317be19a2f2327703f059dbd790e4f28")
DEFAULT_HF_USER = "Mohamad-I8"
SAMPLE_RATE = 24000

KAGGLE_WORKING = "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd()

user_sessions = {}

def get_session(uid):
    if uid not in user_sessions:
        user_sessions[uid] = {"hf_token": DEFAULT_HF_TOKEN, "step": "IDLE"}
    return user_sessions[uid]

def get_hf_api(session):
    token = session.get("hf_token", DEFAULT_HF_TOKEN)
    return HfApi(token=token)

def get_hf_username(api):
    try:
        info = api.whoami()
        return info.get("name", DEFAULT_HF_USER)
    except Exception:
        return DEFAULT_HF_USER

def clean_repo_name(name):
    clean = re.sub(r"^[^\w/-]+|[^\w/-]+$", "", name.strip())
    return clean

def is_valid_audio(fp):
    try: return os.path.exists(fp) and os.path.getsize(fp) >= 200 and sf.info(fp).duration >= 0.1 and sf.info(fp).frames > 0
    except Exception: return False

def convert_to_wav(src, dst, target_sr=SAMPLE_RATE):
    try:
        import librosa
        wav, _ = librosa.load(src, sr=target_sr, mono=True)
        if len(wav) > 0 and np.isfinite(wav).all():
            sf.write(dst, wav, target_sr, subtype="PCM_16")
            if is_valid_audio(dst): return True
    except Exception: pass
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(src).set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
        audio.export(dst, format="wav")
        if is_valid_audio(dst): return True
    except Exception: pass
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16", str(dst)], check=True, capture_output=True)
        if is_valid_audio(dst): return True
    except Exception: pass
    return False

def convert_single_audio(args):
    idx, src_audio, wav_dir, target_sr, t_map = args
    clean_id = f"aseel_{idx+1:06d}"
    dst_wav = os.path.join(wav_dir, f"{clean_id}.wav")
    text = t_map.get(Path(src_audio).stem, Path(src_audio).stem)
    if convert_to_wav(src_audio, dst_wav, target_sr=target_sr):
        return (clean_id, text)
    elif os.path.exists(dst_wav):
        try: os.unlink(dst_wav)
        except Exception: pass
    return None

def parse_transcript_csv(csv_path):
    t_map = {}
    delims = ["|", ",", ";", "\t"]
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines: return t_map
        delim = ","
        for d in delims:
            if d in lines[0]: delim = d; break
        start_idx = 0
        headers = [p.strip().lower() for p in lines[0].split(delim)]
        if any(h in headers for h in ["file_name", "id", "file_id", "text", "transcript", "sentence"]):
            start_idx = 1
        for line in lines[start_idx:]:
            parts = line.split(delim)
            if len(parts) >= 2:
                fid = Path(parts[0].strip()).stem
                txt = parts[-1].strip().strip('"')
                if fid and txt: t_map[fid] = txt
    except Exception: pass
    return t_map

def make_progress_bar(percent, width=12):
    filled = max(0, min(width, int(width * percent / 100)))
    return "█" * filled + "░" * (width - filled)

def build_progress_callback(msg, title_prefix):
    last_update = [0]
    start_time = time.time()

    async def progress_callback(current, total):
        now = time.time()
        if now - last_update[0] < 1.0 and current < total:
            return
        last_update[0] = now
        pct = (current / total) * 100 if total else 0
        bar = make_progress_bar(pct)
        mb_curr = current / (1024 * 1024)
        mb_tot = total / (1024 * 1024)
        
        elapsed = now - start_time
        speed = (mb_curr / elapsed) if elapsed > 0 else 0
        
        text = (
            f" {title_prefix}\n\n"
            f"[{bar}] {pct:.1f}%\n"
            f" الحجم: {mb_curr:.2f} MB / {mb_tot:.2f} MB\n"
            f" السرعة: {speed:.2f} MB/s"
        )
        try:
            await msg.edit_text(text)
        except Exception:
            pass

    return progress_callback

async def fast_upload_dataset_folder(api, folder_path, repo_id, hf_token, msg):
    await msg.edit_text(f" جاري تهيئة رفع البيانات المقاطع إلى Hugging Face ({repo_id})...")
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=hf_token)
    
    start_time = time.time()
    
    def _do_upload():
        return api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="dataset",
            delete_patterns="*",
            token=hf_token
        )

    upload_task = asyncio.create_task(asyncio.to_thread(_do_upload))
    
    dots = [".", "..", "...", "...."]
    idx = 0
    while not upload_task.done():
        await asyncio.sleep(3)
        elapsed = int(time.time() - start_time)
        try:
            await msg.edit_text(
                f" جاري رفع الملفات والمقاطع بسرعة عالية إلى Hugging Face{dots[idx % len(dots)]}\n\n"
                f" المستودع: `{repo_id}`\n"
                f" الوقت المنقضي: {elapsed} ثانية\n\n"
                f" تم تفعيل الرفع التوازي الذكي لضمان أعلى سرعة وعدم التوقف."
            )
        except Exception:
            pass
        idx += 1

    return await upload_task

app = Client("kaggle_optispeech_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN, workdir=KAGGLE_WORKING)

def main_menu_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(" عرض مستودعات الحساب", callback_data="list_repos")],
        [InlineKeyboardButton(" إنشاء مستودع جديد", callback_data="create_repo_start")],
        [InlineKeyboardButton(" رفع داتا سيت جديدة", callback_data="upload_data_start")],
        [InlineKeyboardButton(" تغيير توكن Hugging Face", callback_data="change_token_start")]
    ])

@app.on_message(filters.command(["start", "menu"]) & filters.private)
async def start_cmd(client, message):
    uid = message.from_user.id
    session = get_session(uid)
    session["step"] = "MAIN_MENU"
    await message.reply_text(
        " أهلاً بك في بوت إدارة ورفع البيانات على Hugging Face!\nاختر إحدى الخدمات التالية من القائمة:",
        reply_markup=main_menu_keyboard()
    )

@app.on_message(filters.command("cancel") & filters.private)
async def cancel_cmd(client, message):
    uid = message.from_user.id
    session = get_session(uid)
    session["step"] = "MAIN_MENU"
    await message.reply_text("تم إلغاء العملية والعودة للقائمة الرئيسية.", reply_markup=main_menu_keyboard())

@app.on_callback_query()
async def handle_callbacks(client, callback_query: CallbackQuery):
    uid = callback_query.from_user.id
    data = callback_query.data
    session = get_session(uid)
    api = get_hf_api(session)
    username = get_hf_username(api)

    if data == "main_menu":
        session["step"] = "MAIN_MENU"
        await callback_query.message.edit_text(
            " القائمة الرئيسية لإدارة المستودعات والداتا سيت:",
            reply_markup=main_menu_keyboard()
        )
    elif data == "change_token_start":
        session["step"] = "WAIT_TOKEN"
        kb = ReplyKeyboardMarkup([["استخدام التوكن الافتراضي"]], resize_keyboard=True, one_time_keyboard=True)
        await callback_query.message.reply_text(
            "أدخل توكن Hugging Face الخاص بك (بصلاحيات Write) أو اضغط على الزر لاستخدام التوكن الافتراضي:",
            reply_markup=kb
        )
    elif data == "list_repos":
        await callback_query.message.edit_text(" جاري جلب المستودعات الخاصة بحسابك...")
        try:
            repos = list(api.list_datasets(author=username))
            if not repos:
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton(" إنشاء مستودع جديد", callback_data="create_repo_start")],
                    [InlineKeyboardButton(" القائمة الرئيسية", callback_data="main_menu")]
                ])
                await callback_query.message.edit_text("لم يتم العثور على أية مستودعات بيانات (Datasets) في حسابك.", reply_markup=kb)
                return

            buttons = []
            for r in repos:
                repo_id = r.id
                buttons.append([InlineKeyboardButton(f" {repo_id}", callback_data=f"show_repo:{repo_id}")])
            buttons.append([InlineKeyboardButton(" إنشاء مستودع جديد", callback_data="create_repo_start")])
            buttons.append([InlineKeyboardButton(" القائمة الرئيسية", callback_data="main_menu")])

            await callback_query.message.edit_text(
                f" لديك ({len(repos)}) مستودع/مستودعات بيانات على Hugging Face:\nاضغط على أي مستودع لإدارته أو تعديله أو الرفع عليه:",
                reply_markup=InlineKeyboardMarkup(buttons)
            )
        except Exception as e:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton(" القائمة الرئيسية", callback_data="main_menu")]])
            await callback_query.message.edit_text(f" حدث خطأ أثناء جلب المستودعات: {e}", reply_markup=kb)

    elif data.startswith("show_repo:"):
        repo_id = data.split("show_repo:", 1)[1]
        session["repo_name"] = repo_id
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(" رفع بيانات (CSV + ZIP) إليه", callback_data=f"select_repo_for_upload:{repo_id}")],
            [InlineKeyboardButton(" تعديل اسم المستودع", callback_data=f"rename_repo_start:{repo_id}")],
            [InlineKeyboardButton(" حذف المستودع", callback_data=f"delete_repo_confirm:{repo_id}")],
            [InlineKeyboardButton(" العودة للقائمة", callback_data="list_repos")]
        ])
        await callback_query.message.edit_text(
            f" المستودع المحدد:\n`{repo_id}`\n\nالرابط: https://huggingface.co/datasets/{repo_id}\n\nاختر الإجراء الذي تريد القيام به:",
            reply_markup=kb
        )

    elif data == "create_repo_start":
        session["step"] = "WAIT_CREATE_REPO_NAME"
        await callback_query.message.edit_text(
            f" أدخل اسم المستودع الجديد المراد إنشاؤه لـ ({username}):\n(مثال: `Arabic_Speech_Dataset_v2`)"
        )

    elif data.startswith("rename_repo_start:"):
        target_repo = data.split("rename_repo_start:", 1)[1]
        session["target_repo"] = target_repo
        session["step"] = "WAIT_RENAME_REPO_NAME"
        await callback_query.message.edit_text(
            f" أدخل الاسم الجديد للمستودع الحالي (`{target_repo}`):\nمثال: `New_Dataset_Name`"
        )

    elif data.startswith("delete_repo_confirm:"):
        target_repo = data.split("delete_repo_confirm:", 1)[1]
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(" نعم، احذف المستودع نهائياً", callback_data=f"do_delete_repo:{target_repo}")],
            [InlineKeyboardButton(" إلغاء", callback_data=f"show_repo:{target_repo}")]
        ])
        await callback_query.message.edit_text(
            f" هل أنت متاكد من حذف المستودع التالي من Hugging Face؟\n`{target_repo}`\n\nتنبيه: هذا الإجراء لا يمكن التراجع عنه!",
            reply_markup=kb
        )

    elif data.startswith("do_delete_repo:"):
        target_repo = data.split("do_delete_repo:", 1)[1]
        await callback_query.message.edit_text(f" جاري حذف المستودع `{target_repo}`...")
        try:
            api.delete_repo(repo_id=target_repo, repo_type="dataset")
            kb = InlineKeyboardMarkup([[InlineKeyboardButton(" عرض باقي المستودعات", callback_data="list_repos")]])
            await callback_query.message.edit_text(f" تم حذف المستودع `{target_repo}` بنجاح!", reply_markup=kb)
        except Exception as e:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton(" العودة للمستودع", callback_data=f"show_repo:{target_repo}")]])
            await callback_query.message.edit_text(f" حدث خطأ أثناء حذف المستودع: {e}", reply_markup=kb)

    elif data.startswith("select_repo_for_upload:"):
        repo_id = data.split("select_repo_for_upload:", 1)[1]
        session["repo_name"] = repo_id
        session["step"] = "WAIT_CSV"
        await callback_query.message.edit_text(
            f" تم اختيار المستودع: `{repo_id}`\n\nالآن قم بإرسال ملف النصوص بصيغة (CSV أو TXT)."
        )

    elif data == "upload_data_start":
        session["step"] = "WAIT_REPO"
        await callback_query.message.edit_text(
            f"أدخل اسم المستودع المراد رفع الداتا سيت إليه (مثال: `{username}/Aseel_Arabic_Dataset`):"
        )

@app.on_message(filters.text & filters.private)
async def handle_text(client, message):
    uid = message.from_user.id
    session = get_session(uid)
    step = session.get("step", "MAIN_MENU")
    txt = message.text.strip()
    api = get_hf_api(session)
    username = get_hf_username(api)

    if step == "WAIT_TOKEN":
        token_val = DEFAULT_HF_TOKEN if (txt == "استخدام التوكن الافتراضي" or not txt) else txt
        session["hf_token"] = token_val
        session["step"] = "MAIN_MENU"
        await message.reply_text(" تم حفظ التوكن بنجاح!", reply_markup=ReplyKeyboardRemove())
        await message.reply_text("اختر إحدى الخدمات من القائمة الرئيسية:", reply_markup=main_menu_keyboard())

    elif step == "WAIT_CREATE_REPO_NAME":
        raw_name = clean_repo_name(txt)
        if "/" not in raw_name:
            repo_id = f"{username}/{raw_name}"
        else:
            repo_id = raw_name
        
        msg = await message.reply_text(f" جاري إنشاء المستودع `{repo_id}` على Hugging Face...")
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=session.get("hf_token", DEFAULT_HF_TOKEN))
            session["repo_name"] = repo_id
            session["step"] = "WAIT_CSV"
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton(" رفع بيانات الآن إلى هذا المستودع", callback_data=f"select_repo_for_upload:{repo_id}")],
                [InlineKeyboardButton(" القائمة الرئيسية", callback_data="main_menu")]
            ])
            await msg.edit_text(f" تم إنشاء المستودع `{repo_id}` بنجاح!\n\nيمكنك الآن رفع البيانات إليه مباشرة أو العودة للقائمة الرئيسية.", reply_markup=kb)
        except Exception as e:
            session["step"] = "MAIN_MENU"
            await msg.edit_text(f" فشل إنشاء المستودع: {e}", reply_markup=main_menu_keyboard())

    elif step == "WAIT_RENAME_REPO_NAME":
        old_repo = session.get("target_repo")
        raw_name = clean_repo_name(txt)
        if "/" not in raw_name:
            new_repo = f"{username}/{raw_name}"
        else:
            new_repo = raw_name

        msg = await message.reply_text(f" جاري تغيير اسم المستودع من `{old_repo}` إلى `{new_repo}`...")
        try:
            api.move_repo(from_id=old_repo, to_id=new_repo, repo_type="dataset")
            session["step"] = "MAIN_MENU"
            kb = InlineKeyboardMarkup([[InlineKeyboardButton(" عرض جميع المستودعات", callback_data="list_repos")]])
            await msg.edit_text(f" تم تغيير اسم المستودع بنجاح إلى `{new_repo}`!", reply_markup=kb)
        except Exception as e:
            session["step"] = "MAIN_MENU"
            await msg.edit_text(f" حدث خطأ أثناء نقل/تغيير اسم المستودع: {e}", reply_markup=main_menu_keyboard())

    elif step == "WAIT_REPO":
        repo_name = clean_repo_name(txt)
        if "/" not in repo_name:
            repo_name = f"{username}/{repo_name}"
        session["repo_name"] = repo_name
        session["step"] = "WAIT_CSV"
        await message.reply_text(f"المستودع المستهدف: `{repo_name}`\nالآن أرسل ملف النصوص (CSV).")

    else:
        await message.reply_text("أرسل /start أو /menu للوصول للقائمة الرئيسية.", reply_markup=main_menu_keyboard())

@app.on_message(filters.document & filters.private)
async def handle_document(client, message):
    uid = message.from_user.id
    session = get_session(uid)
    step = session.get("step")
    doc = message.document

    if not step or step not in ["WAIT_CSV", "WAIT_ZIP"]:
        await message.reply_text("يرجى اختيار مستودع أولاً من القائمة الرئيسية عبر /menu")
        return

    if step == "WAIT_CSV":
        if not doc.file_name.lower().endswith((".csv", ".txt")):
            await message.reply_text("يرجى إرسال ملف بصيغة .csv أو .txt")
            return
        work_dir = Path(os.path.join(KAGGLE_WORKING, f"tmp_{uid}"))
        work_dir.mkdir(parents=True, exist_ok=True)
        csv_path = work_dir / "transcripts.csv"
        msg = await message.reply_text(" جاري بدء تحميل ملف النصوص...")
        progress_cb = build_progress_callback(msg, "تحميل ملف النصوص (CSV)")
        await client.download_media(doc, file_name=str(csv_path), progress=progress_cb)
        session["transcript_map"] = parse_transcript_csv(str(csv_path))
        session["step"] = "WAIT_ZIP"
        await msg.edit_text(f" تم استلام ملف النصوص بنجاح ({len(session['transcript_map'])} نص).\n\nالآن أرسل الملف المضغوط (.zip) الخاص بالصوتيات.")

    elif step == "WAIT_ZIP":
        if not doc.file_name.lower().endswith(".zip"):
            await message.reply_text("يرجى إرسال ملف بصيغة .zip")
            return
        work_dir = Path(os.path.join(KAGGLE_WORKING, f"tmp_{uid}"))
        zip_path = work_dir / "input_dataset.zip"
        extracted_dir = work_dir / "extracted"
        raw_dataset_dir = work_dir / "raw_dataset"
        wav_dir = raw_dataset_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        msg = await message.reply_text(" جاري بدء تحميل الملف المضغوط...")
        progress_cb = build_progress_callback(msg, "تحميل الملف المضغوط (.zip)")
        try:
            await client.download_media(doc, file_name=str(zip_path), progress=progress_cb)
            await msg.edit_text(" تم التحميل بنجاح. جاري فك الضغط وفحص المقاطع...")
            with zipfile.ZipFile(zip_path, "r") as z: z.extractall(extracted_dir)
            audio_files = [os.path.join(r, f) for r, _, fs in os.walk(extracted_dir) for f in fs if Path(f).suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".m4a")]
            if not audio_files:
                await msg.edit_text("لم يتم العثور على أية ملفات صوتية داخل الأرشيف.")
                shutil.rmtree(work_dir, ignore_errors=True)
                session["step"] = "MAIN_MENU"
                return
            t_map = session.get("transcript_map", {})
            converted, valid_entries = 0, []
            total_files = len(audio_files)
            
            max_workers = min(16, (os.cpu_count() or 4) * 2)
            tasks_args = [(idx, src, str(wav_dir), SAMPLE_RATE, t_map) for idx, src in enumerate(audio_files)]
            
            async def _convert_with_progress():
                results = []
                last_time = time.time()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(convert_single_audio, arg) for arg in tasks_args]
                    for idx, fut in enumerate(futures):
                        res = fut.result()
                        if res:
                            results.append(res)
                        now = time.time()
                        if now - last_time >= 1.0 or (idx + 1) == total_files:
                            last_time = now
                            pct = ((idx + 1) / total_files) * 100
                            bar = make_progress_bar(pct)
                            try:
                                await msg.edit_text(
                                    f" جاري تحويل المقاطع الصوتية إلى WAV بالتوازي...\n\n"
                                    f"[{bar}] {pct:.1f}%\n"
                                    f" المقاطع المنجزة: {idx + 1} / {total_files}"
                                )
                            except Exception:
                                pass
                return results

            valid_entries = await _convert_with_progress()
            converted = len(valid_entries)

            if not valid_entries:
                await msg.edit_text("فشل تحويل المقاطع الصوتية.")
                shutil.rmtree(work_dir, ignore_errors=True)
                session["step"] = "MAIN_MENU"
                return
            await msg.edit_text(" جاري إنشاء الفهارس والـ CSV النهائية...")
            random.seed(42)
            random.shuffle(valid_entries)
            with open(raw_dataset_dir / "metadata.csv", "w", encoding="utf-8", newline="\n") as f:
                f.write("file_name,text\n")
                for fid, txt in valid_entries:
                    clean_txt = txt.replace('"', '""').replace("\n", " ").strip()
                    f.write(f'wav/{fid}.wav,"{clean_txt}"\n')
            val_size = max(1, int(len(valid_entries) * 0.05))
            for fname, entries in [("train.csv", valid_entries[val_size:]), ("val.csv", valid_entries[:val_size])]:
                with open(raw_dataset_dir / fname, "w", encoding="utf-8", newline="\n") as f:
                    for fid, txt in entries: f.write(f"{fid}|{txt.replace('|', ' ').strip()}\n")
            with open(raw_dataset_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(f"---\nlanguage:\n- ar\nlicense: mit\ntags:\n- audio\n- speech\n- text-to-speech\n- arabic\n---\n# Arabic Speech Dataset\nContains {converted} 24kHz Mono WAV clips.")
            
            await msg.edit_text(" جاري ضغط المقاطع الصوتية في ملف ZIP واحد (wav.zip)...")
            zip_wav_path = raw_dataset_dir / "wav.zip"
            def _zip_all_wavs():
                with zipfile.ZipFile(zip_wav_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fid, _ in valid_entries:
                        wav_file = wav_dir / f"{fid}.wav"
                        if wav_file.exists():
                            zf.write(wav_file, arcname=f"wav/{fid}.wav")
            
            await asyncio.to_thread(_zip_all_wavs)
            shutil.rmtree(wav_dir, ignore_errors=True)

            repo_id = session["repo_name"]
            hf_token = session.get("hf_token", DEFAULT_HF_TOKEN)
            api = get_hf_api(session)
            
            await fast_upload_dataset_folder(api, raw_dataset_dir, repo_id, hf_token, msg)

            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton(" عرض جميع المستودعات", callback_data="list_repos")],
                [InlineKeyboardButton(" القائمة الرئيسية", callback_data="main_menu")]
            ])
            await msg.edit_text(
                f" تمت العملية بنجاح وبسرعة عالية!\n\n"
                f" المستودع: `{repo_id}`\n"
                f" عدد المقاطع: {converted}\n"
                f" التنسيق: 24000Hz Mono WAV\n"
                f" الرابط: https://huggingface.co/datasets/{repo_id}",
                reply_markup=kb
            )
        except Exception as e:
            kb = InlineKeyboardMarkup([[InlineKeyboardButton(" القائمة الرئيسية", callback_data="main_menu")]])
            await msg.edit_text(f" حدث خطأ أثناء العملية: {e}", reply_markup=kb)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            session["step"] = "MAIN_MENU"

def cleanup_stale_sessions():
    session_patterns = [
        os.path.join(KAGGLE_WORKING, "*.session*"),
        "*.session*"
    ]
    for pattern in session_patterns:
        for f in glob.glob(pattern):
            try: os.remove(f)
            except Exception: pass

async def main():
    if not API_ID or not API_HASH:
        print(" خطأ: TELEGRAM_API_ID أو TELEGRAM_API_HASH مفقودان!")
        return
    try:
        await app.start()
    except Exception as e:
        err_str = str(e).lower()
        if "404" in err_str or "auth" in err_str or "key" in err_str or "session" in err_str:
            print("جاري تنظيف الجلسة التالفة وإعادة إنشاء مفتاح التشفير...")
            cleanup_stale_sessions()
            await app.start()
        else:
            raise e

    print("Bot is running...")
    await idle()
    await app.stop()

def run_bot():
    print("Starting Kaggle Pyrogram Bot...")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(main())
    else:
        try:
            app.run()
        except Exception as e:
            err_str = str(e).lower()
            if "404" in err_str or "auth" in err_str or "key" in err_str:
                print("جاري حذف الجلسة التالفة وإعادة تشغيل البوت...")
                cleanup_stale_sessions()
                app.run()
            else:
                raise e

if __name__ == "__main__":
    run_bot()