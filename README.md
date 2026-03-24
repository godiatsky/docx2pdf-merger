# DOCX → PDF Merger

Веб-приложение: загружаешь два .docx, получаешь один PDF.
Конвертация через LibreOffice на сервере.

## Деплой на Railway (бесплатно)

1. Зарегистрируйся на https://railway.app (через GitHub)
2. Нажми **New Project → Deploy from GitHub**
3. Создай репозиторий на GitHub и залей все файлы этой папки
4. Railway сам определит nixpacks.toml и установит LibreOffice
5. Через 2-3 минуты получишь ссылку вида `https://xxx.railway.app`

## Структура файлов

```
docx-merger/
├── app.py              # Flask сервер
├── requirements.txt    # Python зависимости
├── nixpacks.toml       # Конфиг Railway (LibreOffice)
└── static/
    └── index.html      # Веб-интерфейс
```

## Локальный запуск (для теста)

```bash
pip install flask pypdf gunicorn
# LibreOffice должен быть установлен
python app.py
# Открой http://localhost:5000
```
