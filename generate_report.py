import sqlite3
import openpyxl
from datetime import datetime

conn = sqlite3.connect('history.db')
cursor = conn.cursor()
cursor.execute("SELECT timestamp, count, image_path FROM requests")
rows = cursor.fetchall()

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "История учета"
ws.append(["Дата и время", "Количество слонов", "Путь к файлу"])

for row in rows:
    ws.append(row)

filename = f"Elephant_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
wb.save(filename)
print(f"Отчет успешно сохранен в {filename}")