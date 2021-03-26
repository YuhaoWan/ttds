import time
from flask import (
    Blueprint, flash, render_template, request
)

bp = Blueprint('search', __name__, url_prefix='/')

from app.algorithm import *

Search = TextSearch()


#@bp.route("/")
#def root():
    #return render_template("index.html")


@bp.route('/', methods=('GET', 'POST'))
def search():
    start_time = time.time()
    Result = ''
    choice = ''
    content = ''
    num = dict()
    if request.method == 'POST':
        content = request.values.get('query')
        # choice = request.values.get("search_method")
        # Result = "search method: " + choice + "\nsearch content: " + content + \
        # "\nsearch content: \n"

        print(content)
        error = None

        if not content:
            error = 'Content Cannot be None.'

        flash(error)
        num = Search.algorithm('keyword', content)
    end_time = time.time()
    print(end_time - start_time)
    return render_template('music.html', num=num)
