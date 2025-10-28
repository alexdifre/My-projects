from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
import os
import time
import unidecode

model_path = os.path.join(os.path.dirname(__file__), "../resources/cc.fr.300.vec")

start_time = time.time()
model = KeyedVectors.load_word2vec_format(model_path, binary=False)
end_time = time.time()

loading_time = end_time - start_time
print(f"The model has been uploaded in {loading_time:.2f} seconds.")

app = Flask(__name__)


@app.route('/similar')
def similar():
    word = request.args.get('word')

    if word is None:
        return jsonify([])

    word = word.lower()
    word_normalized = unidecode.unidecode(word)

    try:
        similar_words = model.most_similar(word, topn=100)
        filtered = []

        for w, sim in similar_words:
            w_lower = w.lower()
            w_normalized = unidecode.unidecode(w_lower)

            if (w_normalized == word_normalized or
                    word_normalized in w_normalized or
                    w_normalized in word_normalized):
                continue

            if 0.6 <= sim <= 0.9:
                filtered.append((w, abs(sim - 0.8)))

        filtered.sort(key=lambda x: x[1])

        result = [w for w, _ in filtered[:3]]

        return jsonify(result)

    except KeyError:
        return jsonify([])


if __name__ == '__main__':
    app.run(port=8081)
