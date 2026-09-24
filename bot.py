#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import calendar
import re
from datetime import datetime
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from telegram import Update, InputFile, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

TOKEN = "8460468128:AAE_zAf8HFhXPyDQ84KuexH-2tLoTDcLafE"
MOON_TEXTURE_PATH = "8k_moon.jpg"
PHASE_SIZE = 800
PI = math.pi
DATE_DOT_RE = re.compile(r"^\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*$")
DATE_DASH_RE = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$")

def normalize(v: float) -> float:
    v -= math.floor(v)
    return v + 1.0 if v < 0 else v

def round2(x: float) -> float:
    return round(x * 100) / 100.0

def is_valid_date(y: int, m: int, d: int) -> bool:
    if m < 1 or m > 12: return False
    if m == 2:
        leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
        return 1 <= d <= (29 if leap else 28)
    if m in (4, 6, 9, 11): return 1 <= d <= 30
    return 1 <= d <= 31

def moon_data(year: int, month: int, day: int) -> dict:
    if not is_valid_date(year, month, day):
        return {"error": "Неверная дата."}
    YY = year - math.floor((12 - month) / 10)
    MM = month + 9
    if MM >= 12: MM -= 12
    K1 = math.floor(365.25 * (YY + 4712))
    K2 = math.floor(30.6 * MM + 0.5)
    K3 = math.floor(((YY / 100) + 49) * 0.75) - 38
    JD = K1 + K2 + day + 59
    if JD > 2299160: JD -= K3
    IP = normalize((JD - 2451550.1) / 29.530588853)
    AG = IP * 29.53
    if   AG <  1.84566: phase = "Новолуние 🌑"
    elif AG <  5.53699: phase = "Растущий серп 🌒"
    elif AG <  9.22831: phase = "Первая четверть 🌓"
    elif AG < 12.91963: phase = "Прибывающая Луна 🌔"
    elif AG < 16.61096: phase = "Полнолуние 🌕"
    elif AG < 20.30228: phase = "Убывающая Луна 🌖"
    elif AG < 23.99361: phase = "Последняя четверть 🌗"
    else:               phase = "Стареющий серп 🌘"
    illum = (1.0 - math.cos(2.0 * PI * IP)) / 2.0
    return {
        "phase": phase,
        "age_days": round2(AG),
        "phase_fraction": IP,
        "illumination_fraction": illum,
    }

def sample_spherical(img: Image.Image, size: int):
    W, H = img.size
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx + 0.5) / size * 2.0 - 1.0
    dy = 1.0 - (yy + 0.5) / size * 2.0
    r2 = dx*dx + dy*dy
    mask = r2 <= 1.0
    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    lon = np.arctan2(dx, nz)
    lat = np.arcsin(dy)
    u = ((lon / (2*np.pi)) + 0.5) * (W - 1)
    v = (0.5 - lat / np.pi) * (H - 1)
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    u1 = np.minimum(u0+1, W-1); v1 = np.minimum(v0+1, H-1)
    fu = u - u0; fv = v - v0
    Ia = arr[v0, u0]; Ib = arr[v0, u1]
    Ic = arr[v1, u0]; Id = arr[v1, u1]
    top = Ia*(1-fu) + Ib*fu
    bot = Ic*(1-fu) + Id*fu
    return top*(1-fv) + bot*fv, mask

def generate_realistic_phase_image(phase_frac: float, size: int = PHASE_SIZE) -> Image.Image:
    orig = Image.open(MOON_TEXTURE_PATH)
    tex_arr, mask = sample_spherical(orig, size)
    yy, xx = np.mgrid[0:size, 0:size]
    nx = (xx + 0.5) / size * 2.0 - 1.0
    ny = 1.0 - (yy + 0.5) / size * 2.0
    nz = np.sqrt(np.clip(1.0 - nx*nx - ny*ny, 0.0, 1.0))
    theta = 2.0 * math.pi * phase_frac
    sun_dir = np.array([math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)
    dot = nx*sun_dir[0] + nz*sun_dir[2]
    softness = 4.0
    lambert = 1.0 / (1.0 + np.exp(-softness * dot))
    ambient_lit, ambient_dark = 0.08, 0.03
    light = np.where(dot > 0, lambert + ambient_lit, ambient_dark)
    light = np.clip(light, 0.0, 1.0)
    light_img = Image.fromarray((light * 255).astype(np.uint8), mode='L')
    light_img = light_img.filter(ImageFilter.GaussianBlur(radius=12))
    light = np.asarray(light_img, dtype=np.float32) / 255.0
    if abs(phase_frac - 0.5) < 0.12:
        boost = 0.22 * (1.0 - abs(phase_frac - 0.5) / 0.12)
        light = np.clip(light + boost, 0.0, 1.0)
    light = np.power(light, 1.0 / 1.6)
    shaded = tex_arr * light
    moon8 = (np.clip(shaded, 0, 1) * 255).astype(np.uint8)
    moon_img = Image.fromarray(moon8, mode="L").convert("RGB")
    out = Image.new("RGB", (size, size), (0, 0, 0))
    alpha = (mask.astype(np.uint8) * 255)
    mask_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(2.5))
    glow = moon_img.filter(ImageFilter.GaussianBlur(8))
    out.paste(glow, (0, 0), mask_img)
    out.paste(moon_img, (0, 0), mask_img)
    out = ImageEnhance.Brightness(out).enhance(1.2)
    out = ImageEnhance.Contrast(out).enhance(1.1)
    return out

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_kb = [["🌙 Сегодня", "📅 Календарь месяца"]]
    markup = ReplyKeyboardMarkup(reply_kb, resize_keyboard=True)

    text = (
        "✨ *Добро пожаловать в Лунный бот!* ✨\n\n"
        "Я помогу узнать:\n"
        "• текущую фазу Луны 🌕\n"
        "• фазу на любую дату 📆\n"
        "• календарь фаз на месяц 🗓️\n\n"
        "Введи дату (`DD.MM.YYYY`, `YYYY-MM-DD`, `today`, `сегодня`) или нажми кнопку ниже."
    )
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=markup)

async def _send_day(msg, y: int, m: int, d: int):
    md = moon_data(y, m, d)
    if "error" in md:
        await msg.reply_text(md["error"])
        return
    illum = round(md["illumination_fraction"] * 100, 1)
    text = (
        f"📅 *{d:02}.{m:02}.{y}*\n"
        f"Фаза: *{md['phase']}*\n"
        f"Возраст: *{md['age_days']}* дн.\n"
        f"Освещённость: *{illum}%*"
    )
    await msg.reply_text(text, parse_mode="Markdown")
    img = generate_realistic_phase_image(md["phase_fraction"], PHASE_SIZE)
    bio = BytesIO(); img.save(bio, format="PNG"); bio.seek(0)
    await msg.reply_photo(photo=InputFile(bio, filename="moon.png"))

def _month_table(y: int, m: int) -> str:
    days = calendar.monthrange(y, m)[1]
    header = f"📅 *Календарь фаз Луны — {m:02}.{y}*\n"
    lines = [header, "```"]
    for dd in range(1, days+1):
        md = moon_data(y, m, dd)
        if "error" in md:
            lines.append(f"{dd:02}.{m:02} | ошибка даты")
            continue
        illum = round(md["illumination_fraction"] * 100, 1)
        age = f"{md['age_days']:>4}"
        lines.append(f"{dd:02}.{m:02} | {md['phase']:<20} | {age} д | {illum:>5}%")
    lines.append("```")
    return "\n".join(lines)

async def text_date_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip().lower()
    if txt in ("🌙 сегодня", "сегодня", "today"):
        now = datetime.utcnow()
        await _send_day(update.message, now.year, now.month, now.day)
        return
    if txt in ("📅 календарь месяца", "календарь месяца", "month", "месяц"):
        now = datetime.utcnow()
        text = _month_table(now.year, now.month)
        await update.message.reply_text(text, parse_mode="Markdown")
        return
    m1 = DATE_DOT_RE.match(txt)
    m2 = DATE_DASH_RE.match(txt)
    if m1:
        d, m, y = map(int, m1.groups())
    elif m2:
        y, m, d = map(int, m2.groups())
    else:
        await update.message.reply_text("Не распознал дату. Используй DD.MM.YYYY или YYYY-MM-DD, либо 'today'.")
        return

    await _send_day(update.message, y, m, d)

async def moon_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if args and args[0].lower() == "month":
        now = datetime.utcnow()
        text = _month_table(now.year, now.month)
        await update.message.reply_text(text, parse_mode="Markdown")
        return
    if not args or args[0].lower() in ("today", "сегодня"):
        now = datetime.utcnow()
        await _send_day(update.message, now.year, now.month, now.day)
        return
    arg = args[0]
    if "." in arg:
        try:
            d, m, y = map(int, arg.split("."))
        except Exception:
            await update.message.reply_text("Формат даты: DD.MM.YYYY")
            return
    elif "-" in arg:
        try:
            y, m, d = map(int, arg.split("-"))
        except Exception:
            await update.message.reply_text("Формат даты: YYYY-MM-DD")
            return
    else:
        await update.message.reply_text("Аргумент не распознан. Используй: today | DD.MM.YYYY | YYYY-MM-DD | month")
        return

    await _send_day(update.message, y, m, d)

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("moon", moon_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_date_handler))
    app.run_polling()

if __name__ == "__main__":
    main()