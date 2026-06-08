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

## Environment variables

Copy `.env.example` to `.env` and fill in your API key:

```
cp .env.example .env
```

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
