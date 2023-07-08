from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SubmitField
import torch
from dcgan_generator import netG, z, device, batch_size
from torchvision.utils import save_image

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
	return render_template('index.html')

@app.route('/generate')
def generate():
	return render_template('generate.html')

@app.route('/generate_image',methods= ['GET', 'POST'])
def generate_image():
	noise = torch.randn(batch_size, z, 1, 1, device=device)
	gen_image = netG.forward(noise)
	fake_fname = 'static/generated_image.png'
	save_image(gen_image, fake_fname)
	return redirect(url_for('generate'))


if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8000, debug=True)
