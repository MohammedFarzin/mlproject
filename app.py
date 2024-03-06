from flask import Flask, request, render_template
import numpy as np
import pandas as pd


application = Falsk(__name__)

## Route for home page
@application.route('/')
def index():
    return render_template('index.html')

