# Preprocessing Functions

def extract_ipq(text):
    if not isinstance(text, str): return np.nan
    text = text.lower()
    patterns = [r'(\d+)\s*[-]?\s*pack', r'pack\s*of\s*(\d+)', r'(\d+)\s*ct', r'(\d+)\s*count',
                r'qty[:\s]+(\d+)', r'(\d+)\s*x\s*\d*', r'(\d+)\s*pcs', r'(\d+)\s*piece']
    for p in patterns:
        m = re.search(p, text)
        if m:
            try: return int(m.group(1))
            except: pass
    m = re.search(r'\b([1-9]\d?)\b', text)
    return int(m.group(1)) if m else np.nan

def text_stats(text):
    if not isinstance(text, str): return 0,0
    toks = re.findall(r'\w+', text)
    return len(toks), len(set(toks))

def link_to_local_path(link):
    if not isinstance(link, str) or len(link.strip())==0: return None
    return IMAGES_DIR / Path(link).name

def smape(y_true, y_pred):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    den[den == 0] = 1e-9
    return np.mean(num / den) * 100.0
