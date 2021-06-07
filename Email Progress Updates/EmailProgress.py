import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import pandas as pd

import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import emailprogressconfig as config
import datetime
import time
import numpy as np
import os

class EmailProgress(keras.callbacks.Callback):
    
    def __init__(self, epochs, name):
        self.modelname = name
        self.epochs = epochs
        self.last_epoch_time = 0.0
        self.context = ssl.create_default_context()
        self.training_df = pd.DataFrame(columns=['Epoch Number', 'Loss', 'Accuracy', 'Validation Loss', 'Validation Accuracy'])

    def on_train_begin(self, logs=None):
        subject = f"{self.modelname} Training Started at {datetime.datetime.now()}"
        message = f"""<h1>Prediction model training has started at {datetime.datetime.now()}, running for {self.epochs} epochs</h1>
        <p>Updates will be sent at 25% (epoch {self.epochs // 4}), 50% (epoch {self.epochs // 2}), 75% (epoch {(self.epochs // 4) * 3}) and 100% (epoch {self.epochs})</p>"""
        self.send_email(subject, message, None)

    def on_train_end(self, logs=None):
        N = self.epochs
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.training_df["Loss"], label="train_loss")
        plt.plot(self.training_df["Validation Loss"], label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="center right")

        if os.path.isfile('loss.jpg'):
            os.remove('loss.jpg')

        plt.savefig('loss.jpg')

        N = self.epochs
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.training_df["Accuracy"], label="train_acc")
        plt.plot(self.training_df["Validation Accuracy"], label="val_acc")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="center right")

        if os.path.isfile('accuracy.jpg'):
            os.remove('accuracy.jpg')

        plt.savefig('accuracy.jpg')
        subject = f"{self.modelname} Training Ended at {datetime.datetime.now()}"
        message = f"""
            <h1>Model training has ended at {datetime.datetime.now()}</h1>
            <h2>Last 10 training rows can be seen below.</h2>
            {self.training_df.tail(10).to_html()}
            <h2>Accuracy Graph After {self.epochs} Epochs</h2>
            <img src="cid:accuracy"></img>
            <h2>Loss Graph After {self.epochs} Epochs</h2>
            <img src="cid:loss"></img>
            """
        self.send_email(subject, message, "end")

    def on_epoch_begin(self, epoch, logs):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.last_epoch_time = (time.time() - self.epoch_time_start)
        row = {'Epoch Number':epoch, 'Loss':logs['loss'], 'Accuracy':logs['accuracy'], 'Validation Loss':logs['val_loss'], 'Validation Accuracy':logs['val_accuracy']}
        self.training_df = self.training_df.append(row, ignore_index=True)
        percentage, send = self.progress_check(epoch)
        if send == True:
            N = self.epochs - (epoch + 1)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(self.training_df["Loss"], label="train_loss")
            plt.plot(self.training_df["Validation Loss"], label="val_loss")
            plt.title("Training Loss")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend(loc="center right")

            if os.path.isfile('loss.jpg'):
                os.remove('loss.jpg')

            plt.savefig('loss.jpg')

            N = self.epochs - (epoch + 1)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(self.training_df["Accuracy"], label="train_acc")
            plt.plot(self.training_df["Validation Accuracy"], label="val_acc")
            plt.title("Training Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend(loc="center right")

            if os.path.isfile('accuracy.jpg'):
                os.remove('accuracy.jpg')

            plt.savefig('accuracy.jpg')
            
            subject = f"{self.modelname} Training At {percentage}%"
            message = f"""
            <h1>Model training has reached {percentage}% completion</h1>
            <h2>Last 3 training rows can be seen below.</h2>
            {(self.training_df.tail(3)).to_html()}
            <h2>Accuracy Graph After {epoch} Epochs</h2>
            <img src="cid:accuracy"></img>
            <h2>Loss Graph After {epoch} Epochs</h2>
            <img src="cid:loss"></img>
            <br>
            <br>
            Model expected to complete training in {str(datetime.timedelta(seconds=self.last_epoch_time * (self.epochs - epoch)))}.
            """
            self.send_email(subject, message, "progress")

    def progress_check(self, epoch_num):
        progress_percentage = epoch_num / self.epochs * 100
        if progress_percentage == 25.0 or progress_percentage == 50.0 or progress_percentage == 75.0:
            send = True
        else:
            send = False

        return progress_percentage, send

    def send_email(self, subject, message, category):
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.SENDER_EMAIL
        msg['To'] = config.RECIEVER_EMAIL

        html = f"""\
        <html>
            <head></head>
            <body>
                {message}
            </body>
        </html>
        """

        mainbody = MIMEText(html, 'html')
        msg.attach(mainbody)

        if category == "progress" or category == "end":
            fp = open('accuracy.jpg', 'rb')
            accImage = MIMEImage(fp.read())
            fp.close()

            fp = open('loss.jpg', 'rb')
            lossImage = MIMEImage(fp.read())
            fp.close()

            accImage.add_header('Content-ID', '<accuracy>')
            msg.attach(accImage)

            lossImage.add_header('Content-ID', '<loss>')
            msg.attach(lossImage)

        smtp = smtplib.SMTP_SSL(config.EMAIL_SERVER, config.PORT, context=self.context)
        smtp.login(config.SENDER_EMAIL, config.PASSWORD)
        smtp.sendmail(config.SENDER_EMAIL, config.RECIEVER_EMAIL, msg.as_string())
