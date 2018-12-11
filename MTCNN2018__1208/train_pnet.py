import nets
import train
import os

if __name__ == '__main__':
    net = nets.PNet()

    if not os.path.exists("param"):
        os.makedirs("param")

    trainer = train.Trainer(net, 'param/pnet.pt', r"E:\celeba1208\12")
    trainer.train()
