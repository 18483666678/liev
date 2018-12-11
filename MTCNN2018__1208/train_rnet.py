import nets
import train
import os


if __name__ == '__main__':
    net = nets.RNet()
    if not os.path.exists("param"):
        os.makedirs("param")

    trainer = train.Trainer(net,"param/rnet.pt",r"E:\celeba1208\24")
    trainer.train()