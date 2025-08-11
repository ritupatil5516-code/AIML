import re
from datetime import datetime
MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,"august":8,"september":9,"october":10,"november":11,"december":12,"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}
def parse_month(text: str):
    t = text.lower()
    m = re.search(r'(20\d{2})[-/](0?[1-9]|1[0-2])', t)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r'(0?[1-9]|1[0-2])[-/](20\d{2})', t)
    if m: return int(m.group(2)), int(m.group(1))
    m = re.search(r'(?P<mon>[a-z]{3,9})\s*(?P<yr>20\d{2})', t)
    if m and m.group('mon') in MONTHS: return int(m.group('yr')), MONTHS[m.group('mon')]
    for name, num in MONTHS.items():
        if re.search(rf'\b{name}\b', t): return datetime.utcnow().year, num
    m = re.search(r'(?P<mon>[a-z]{3,9})[-](?P<yy>\d{2})', t)
    if m and m.group('mon') in MONTHS: return 2000+int(m.group('yy')), MONTHS[m.group('mon')]
    return None, None
def month_key(dt_iso: str) -> str:
    return dt_iso[:7]


def parse_last_n_months(text: str) -> int | None:
    t = text.lower()
    m = re.search(r'last\s+(\d{1,2})\s+months?', t)
    if m:
        n = int(m.group(1))
        return max(1, min(24, n))  # cap 1..24
    if 'last six months' in t or 'last 6 months' in t:
        return 6
    return None
