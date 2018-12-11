import nets
import train
import os


if __name__ == '__main__':

    net = nets.ONet()
    if not os.path.exists("param"):
        os.makedirs("param")

    trainer = train.Trainer(net,"param/onet.pt",r"E:\celeba1208\48")
    trainer.train()