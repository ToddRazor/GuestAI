import os
import time
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from ultralytics import YOLO


app = Flask(__name__)

# Папки проекта
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = os.path.join("static", "results")
DATABASE_NAME = "history.db"

# Создание папок, если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Загрузка предобученной модели YOLOv8
# yolov8s.pt точнее, чем yolov8n.pt, поэтому оставляем ее
model = YOLO("yolov8s.pt")


def init_database():
    """
    Создание базы данных SQLite и таблицы history.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            result_filename TEXT NOT NULL,
            people_count INTEGER NOT NULL,
            processing_time REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_history(original_filename, result_filename, people_count, processing_time):
    """
    Сохранение информации об обработке изображения в базу данных.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history (
            timestamp,
            original_filename,
            result_filename,
            people_count,
            processing_time
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        original_filename,
        result_filename,
        people_count,
        processing_time
    ))

    conn.commit()
    conn.close()


def get_history():
    """
    Получение истории обработанных изображений из базы данных.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, timestamp, original_filename, result_filename, people_count, processing_time
        FROM history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    history = []

    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "original_filename": row[2],
            "result_filename": row[3],
            "people_count": row[4],
            "processing_time": row[5]
        })

    return history


def clear_folder(folder_path):
    """
    Удаление всех файлов из указанной папки.
    """
    if not os.path.exists(folder_path):
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)


@app.route("/")
def index():
    """
    Главная страница веб-приложения.
    """
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    """
    Прием изображения, обработка через YOLOv8,
    подсчет людей и сохранение результата.
    """
    if "image" not in request.files:
        return jsonify({"error": "Файл изображения не найден"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    start_time = time.time()

    file_bytes = file.read()

    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Не удалось прочитать изображение"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_filename = f"original_{timestamp}.jpg"
    result_filename = f"result_{timestamp}.jpg"

    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    # Сохраняем исходное изображение
    cv2.imwrite(original_path, image)

    # Запускаем YOLOv8.
    # classes=[0] означает, что модель ищет только людей.
    # В наборе COCO класс person имеет номер 0.
    # conf=0.35 выбран экспериментально как компромиссный порог уверенности.
    # iou=0.45 помогает уменьшить количество дублирующихся рамок.
    results = model(image, classes=[0], conf=0.35, iou=0.45)

    result = results[0]

    # Так как мы указали classes=[0], все найденные объекты являются людьми.
    if result.boxes is not None:
        people_count = len(result.boxes)
    else:
        people_count = 0

    # Отрисовка рамок на изображении
    output_image = result.plot()

    # Сохраняем обработанное изображение
    cv2.imwrite(result_path, output_image)

    processing_time = round(time.time() - start_time, 3)

    # Сохраняем запись в историю
    save_history(
        original_filename=original_filename,
        result_filename=result_filename,
        people_count=people_count,
        processing_time=processing_time
    )

    return jsonify({
        "people_count": people_count,
        "processing_time": processing_time,
        "result_image": "/" + result_path.replace("\\", "/")
    })


@app.route("/history")
def history():
    """
    Возврат истории обработок в формате JSON.
    """
    return jsonify(get_history())


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """
    Очистка истории обработок, исходных изображений,
    обработанных изображений и временных Excel-файлов.
    """
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # Очищаем таблицу истории
        cursor.execute("DELETE FROM history")

        # Сбрасываем счетчик ID, чтобы после очистки записи снова начинались с 1
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='history'")

        conn.commit()
        conn.close()

        # Удаляем исходные изображения
        clear_folder(UPLOAD_FOLDER)

        # Удаляем обработанные изображения
        clear_folder(RESULT_FOLDER)

        # Удаляем временные Excel-файлы, если они есть
        temporary_reports = [
            "report.xlsx",
            "excel_report.xlsx",
            "Отчет_по_результатам_подсчета_посетителей.xlsx"
        ]

        for report_file in temporary_reports:
            if os.path.exists(report_file):
                os.remove(report_file)

        return jsonify({
            "message": "История обработок и сохраненные изображения очищены."
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


@app.route("/export_excel")
def export_excel():
    """
    Формирование Excel-отчета по истории обработок изображений.
    """
    history_data = get_history()

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "История обработок"

    headers = [
        "ID",
        "Дата и время",
        "Исходный файл",
        "Результирующий файл",
        "Количество посетителей",
        "Время обработки, сек."
    ]

    sheet.append(headers)

    # Оформление заголовков
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    # Заполнение таблицы данными
    for item in history_data:
        sheet.append([
            item["id"],
            item["timestamp"],
            item["original_filename"],
            item["result_filename"],
            item["people_count"],
            item["processing_time"]
        ])

    # Настройка ширины столбцов
    sheet.column_dimensions["A"].width = 8
    sheet.column_dimensions["B"].width = 22
    sheet.column_dimensions["C"].width = 32
    sheet.column_dimensions["D"].width = 32
    sheet.column_dimensions["E"].width = 25
    sheet.column_dimensions["F"].width = 24

    # Выравнивание данных
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(horizontal="center")

    # Техническое имя файла на сервере
    report_path = "excel_report.xlsx"

    # Название файла, которое увидит пользователь при скачивании
    download_filename = "Отчет_по_результатам_подсчета_посетителей.xlsx"

    workbook.save(report_path)

    return send_file(
        report_path,
        as_attachment=True,
        download_name=download_filename
    )


if __name__ == "__main__":
    init_database()
    app.run(debug=True)