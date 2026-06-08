# Setup

## Python dependencies

```
pip install -r requirements.txt
```

## System dependencies

### Poppler (required for OCR fallback)

**Windows**
1. Download the latest release from https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to a folder (e.g. `C:\poppler\`)
3. Add `C:\poppler\Library\bin` to system PATH via:
   - Start → search "environment variables" → System Properties → Environment Variables
   - Under "System variables" select `Path` → Edit → New → paste the path

**macOS**
```
brew install poppler
```

**Linux**
```
apt install poppler-utils
```

### Tesseract (required for OCR fallback)

**Windows**
1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer — check "Add to PATH" during setup
3. Or add manually: Start → "environment variables" → System variables → `Path` → New → paste the install path (e.g. `D:\Tesseract_OCR`)

**macOS**
```
brew install tesseract
```

**Linux**
```
apt install tesseract-ocr
```

## Environment variables

Copy `.env.example` to `.env` and fill in your API key:

```
cp .env.example .env
```

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
