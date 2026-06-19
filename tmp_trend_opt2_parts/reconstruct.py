from pathlib import Path
import base64
parts = sorted(Path('tmp_trend_opt2_parts').glob('part*.b64'))
payload = ''.join(p.read_text(encoding='utf-8').strip() for p in parts)
Path('tmp_qqq_trend_optimizer_v2.py').write_bytes(base64.b64decode(payload))
