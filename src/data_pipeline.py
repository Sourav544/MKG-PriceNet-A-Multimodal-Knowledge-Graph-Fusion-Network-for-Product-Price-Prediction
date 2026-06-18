# Data Loading
# ==========================================
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print("Train:", train.shape, "Test:", test.shape)

for df in (train, test):
    df['catalog_content'] = df['catalog_content'].fillna("").astype(str)
    df['catalog_len'] = df['catalog_content'].apply(len)
    df['catalog_nwords'], df['catalog_nuniq'] = zip(*df['catalog_content'].apply(text_stats))
    df['ipq'] = df['catalog_content'].apply(extract_ipq)

train['price'] = train['price'].astype(float)
train['log_price'] = np.log1p(train['price'])

# ==========================================
#  Feature Extraction (Text & Image)
# ==========================================
# Assume extract_text_embeddings and extract_clip_image_embeddings are defined 
# exactly as in your original code (omitted here for brevity, but you keep them in your notebook).

if TEXT_EMB_CACHE.exists():
    print("Loading cached text embeddings...")
    text_emb_all = np.load(TEXT_EMB_CACHE)
else:
    raise FileNotFoundError("Run your Mistral extraction cell first to build TEXT_EMB_CACHE.")

if IMAGE_EMB_CACHE.exists():
    print("Loading cached image embeddings...")
    image_emb_all = np.load(IMAGE_EMB_CACHE)
else:
    raise FileNotFoundError("Run your CLIP extraction cell first to build IMAGE_EMB_CACHE.")