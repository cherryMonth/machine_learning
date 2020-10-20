from tornado.options import options, define
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from app import create_app
from flask import render_template
from app.form import Login

import gensim
from document_similarity import getKeywords, word2vec, simlarityCalu

app = create_app()
model_file = 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

@app.route("/", methods=['GET', 'POST'])
def index():
    form = Login()
    if form.validate_on_submit():
        form.source_keyword.data = getKeywords(form.source.data)
        form.target_keyword.data = getKeywords(form.target.data)
        p1_vec = word2vec(form.source_keyword.data, model)
        p2_vec = word2vec(form.target_keyword.data, model)
        form.score.data = simlarityCalu(p1_vec, p2_vec)
        return render_template("index.html", form=form)
    return render_template("index.html", form=form)


define(name="port", default=5000, type=int)

print ('Server running on http://localhost:%s' % options.port)
http_server = HTTPServer(WSGIContainer(app))
http_server.listen(options.port)
IOLoop.instance().start()
