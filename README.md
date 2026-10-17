# Vision Assistant

Show an object to the camera and OpenAI's vision model (`gpt-4o`) identifies it and gives a short, interesting fact.

## Setup
1. `python -m venv venv` and activate it (Windows: `venv\Scripts\activate`).
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `API_KEY` to your OpenAI key (`sk-...`).
4. `python main.py`

## Keys
SPACE - identify | A - live mode (automatic) | C - clear | Q - quit

## Settings (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `API_KEY` | - | Your OpenAI API key (`OPENAI_API_KEY` also works) |
| `BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `MODEL` | `gpt-4o` | Vision-capable model |
| `CAMERA_INDEX` | `0` | Camera number |
| `LANGUAGE` | `uz` | `uz` or `en` |
| `AUTO_INTERVAL` | `4` | Seconds between requests in live mode |
| `FONT_PATH` | auto | Path to a `.ttf` font if letters show as squares |

## Troubleshooting
- Camera won't open: try `CAMERA_INDEX=1`.
- Squares instead of letters: set `FONT_PATH` to a `.ttf` file.
- Authentication error: check that `API_KEY` is a valid OpenAI key and that your account has billing/credits.
- Never commit your `.env` file; it is listed in `.gitignore`.
