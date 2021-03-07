import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('search', __name__, url_prefix='/')


@bp.route('/', methods=('GET', 'POST'))
def search():
	Result = ''
	choice = ''
	if request.method == 'POST':
		content = request.form['search']
		choice = request.values.get("search_method")
		Result = "search method: "+choice+"\nsearch content: "+content
		# print(Result)
		error = None

		if not content:
			error = 'Content Cannot be None.'

		flash(error)

	return render_template('search.html', Result = Result)

def algorithm(content):
	return content







