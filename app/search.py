from flask_bootstrap import Bootstrap
from flask import Flask, render_template, current_app, request, Blueprint, flash
from .flask_paginate1 import Pagination, get_page_parameter, get_per_page_parameter
import pandas
from app.algorithm import *
import time
import re

bp = Blueprint('search', __name__, url_prefix='/')


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/search', methods=('GET', 'POST'))
def search():
    global query
    query = request.form['keyword']

    start_time = time.time()
    data_dict = algorithm('song', query)
    duration_time = time.time() - start_time

    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    total = len(data_dict)
    pagination_users = get_users(data_dict, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template('search.html',
                           data=pagination_users,
                           page=page,
                           per_page=per_page,
                           pagination=pagination,
                           search_keyword=query,
                           data_num=len(data_dict),
                           time=round(duration_time,2),
                           highlight_word=highlight_word)


@bp.route('/songs', methods=('GET', 'POST'))
def search_Songs():

    # start_time = time.time()
    data_dict = algorithm('song', query)
    # duration_time = time.time() - start_time

    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    total = len(data_dict)
    pagination_users = get_users(data_dict, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template('songs.html',
                           search_keyword=query,
                           data=pagination_users,
                           page=page,
                           per_page=per_page,
                           pagination=pagination,
                           data_num=len(data_dict),
                           highlight_word=highlight_word
                           )


@bp.route('/artists', methods=('GET', 'POST'))
def artists():

    # start_time = time.time()
    data_dict = algorithm('song', query)
    # duration_time = time.time() - start_time

    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    total = len(data_dict)
    pagination_users = get_users(data_dict, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template('artists.html',
                           search_keyword=query,
                           data=pagination_users,
                           page=page,
                           per_page=per_page,
                           pagination=pagination,
						   data_num=len(data_dict),
                           highlight_word=highlight_word
                           )


@bp.route('/albums', methods=('GET', 'POST'))
def albums():

    # start_time = time.time()
    data_dict = algorithm('song', query)
    # duration_time = time.time() - start_time

    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    total = len(data_dict)
    pagination_users = get_users(data_dict, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template('albums.html',
                           search_keyword=query,
                           data=pagination_users,
                           page=page,
                           per_page=per_page,
                           pagination=pagination,
                           data_num=len(data_dict),
                           highlight_word=highlight_word
                           )


@bp.route('/lyrics', methods=('GET', 'POST'))
def search_Lyrics():

    # start_time = time.time()
    data_dict = algorithm('song', query)
    # duration_time = time.time() - start_time

    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    total = len(data_dict)
    pagination_users = get_users(data_dict, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template('lyrics.html',
                           search_keyword=query,
                           data=pagination_users,
                           page=page,
                           per_page=per_page,
                           pagination=pagination,
                           data_num=len(data_dict),
                           highlight_word=highlight_word
                           )


def get_page_args(
        page_parameter=None, per_page_parameter=None, for_test=False, **kwargs
):
    """param order: 1. passed parameter 2. request.args 3: config value
    for_test will return page_parameter and per_page_parameter"""
    args = request.args.copy()
    args.update(request.view_args.copy())

    page_name = get_page_parameter(page_parameter, args)
    per_page_name = get_per_page_parameter(per_page_parameter, args)
    for name in (page_name, per_page_name):
        if name in kwargs:
            args.setdefault(name, kwargs[name])

    if for_test:
        return page_name, per_page_name

    page = int(args.get(page_name, 1, type=int))
    per_page = args.get(per_page_name, type=int)
    if not per_page:
        per_page = int(current_app.config.get(per_page_name.upper(), 20000))  # 这里改per_page
    else:
        per_page = int(per_page)

    offset = (page - 1) * per_page
    return page, per_page, offset


def highlight_word(item, word):
    return re.sub(word, matchCase(word), item, flags=re.IGNORECASE)


def matchCase(word):
    def replace(m):
        text = m.group()
        if text.isupper():
            return '<mark>'+word.upper()+'</mark>'
        elif text.islower():
            return '<mark>'+word.lower()+'</mark>'
        elif text[0].isupper():
            return '<mark>'+word.capitalize()+'</mark>'
        else:
            return '<mark>'+word+'</mark>'
    return replace


def get_users(data_dict, offset=0, per_page=20):
    keys = sorted(data_dict.keys())[offset:offset + per_page]
    return {key: data_dict[key] for key in keys}


