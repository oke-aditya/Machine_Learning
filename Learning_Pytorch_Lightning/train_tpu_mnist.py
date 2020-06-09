import pytorch_lightning as pl
from Lightning_mnist_tpu import mnist_nn

model = mnist_nn()
trainer = pl.Trainer(num_tpu_cores=8, max_epochs=10, progress_bar_refresh_rate=10)
trainer.fit(model)

