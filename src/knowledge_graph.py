# Phase 5: Graph Construction
# ==========================================
def build_and_embed_kg(texts, embeddings, n_neighbors=5, dimensions=64, cache_path=None):
    if cache_path and Path(cache_path).exists():
        print("Loading cached Knowledge Graph embeddings...")
        return np.load(cache_path)
        
    print("Building Semantic KNN Graph...")
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings)
    
    G = nx.Graph()
    n_samples = len(texts)
    G.add_nodes_from(range(n_samples))
    
    for i in tqdm(range(n_samples), desc="Adding Semantic Edges"):
        for j in range(1, n_neighbors): 
            weight = 1.0 - distances[i, j]
            if weight > 0.5:
                G.add_edge(i, indices[i, j], weight=weight)
                
    print("Extracting Ontological Nodes (TF-IDF LCS)...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    
    for i in tqdm(range(n_samples), desc="Adding Attribute Edges"):
        row = X_tfidf.getrow(i).toarray()[0]
        top_indices = row.argsort()[-3:][::-1] 
        for idx in top_indices:
            if row[idx] > 0.1:
                kw_node = f"KW_{vocab[idx]}"
                G.add_edge(i, kw_node, weight=row[idx])
                
    print(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("Running Node2Vec (this may take a few minutes)...")
    
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=15, num_walks=10, workers=8, quiet=False)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)
    
    kg_embeddings = np.zeros((n_samples, dimensions))
    for i in range(n_samples):
        kg_embeddings[i] = model.wv[str(i)]
        
    if cache_path:
        np.save(cache_path, kg_embeddings)
        print(f"Saved KG embeddings to {cache_path}")
        
    return kg_embeddings

combined_text = pd.concat([train['catalog_content'], test['catalog_content']], axis=0).tolist()
kg_emb_all = build_and_embed_kg(combined_text, text_emb_all, dimensions=64, cache_path=KG_EMB_CACHE)

# ==========================================
# Phase 6: Feature Assembly
# ==========================================
train_text_emb, test_text_emb = text_emb_all[:len(train)], text_emb_all[len(train):]
train_img_emb, test_img_emb = image_emb_all[:len(train)], image_emb_all[len(train):]
train_kg_emb, test_kg_emb = kg_emb_all[:len(train)], kg_emb_all[len(train):]

numeric_cols = ['catalog_len','catalog_nwords','catalog_nuniq','ipq']
X_num_train = train[numeric_cols].fillna(0).values
X_num_test = test[numeric_cols].fillna(0).values

scaler = StandardScaler()
X_num_train_s = scaler.fit_transform(X_num_train)
X_num_test_s = scaler.transform(X_num_test)
joblib.dump(scaler, SCALER_FILE)

# Concatenate all modalities
X_train = np.hstack([X_num_train_s, train_text_emb, train_img_emb, train_kg_emb])
X_test = np.hstack([X_num_test_s, test_text_emb, test_img_emb, test_kg_emb])
y_train = train['log_price'].values

print("Fusion dims:", X_train.shape, X_test.shape)
