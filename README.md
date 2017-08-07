# rcnn_losses_visualizations

This codes relies on the log generated by the training process of a Region based CNN (RCNN), which has a different architecture than a normal CNN. Still working on the training accuracy. I am currently using a zf architecture, which has at the top a cls-loss and bbox_loss layer. I will manage to have a training accuracy layer later on.

## Generate training logs

There are many ways out there to do it, but the commom one is to add at the end of you command line for training:

```bash
2>&1 | tee log/detection_log.txt
```
like this:

```bash
time ./tools/train_net.py --gpu 0 --solver ../../solver.prototxt --cfg ../../x.yml --weights ../../zf_iter_20.caffemodel --imdb imagenet_train --iter 20 2>&1 | tee -a log/my_modell.log
```
The __a__ before the location of the file is to append the created the file with incoming data, if the training has been restarted from a pause. 

which basically redirecting the stdout and stderr to detection_log.txt.

### Disclaimer

This code has been run on __ZF__ network. Some names change in the log file. For instance, cls\_loss becomes loss\_cls in a __VGG_1024__.

Original code: http://blog.csdn.net/wxplol/article/details/73694657
