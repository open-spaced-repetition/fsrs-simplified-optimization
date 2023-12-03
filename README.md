# Introduction

FSRS is a time-series model for predicting the memory of spaced repetition users. Its originial version requires BPTT (back propagation through time) to train the model. However, most programming languages don't have mature and efficient deep learning libraries that support BPTT. Therefore, we try to simplify the optimization process of FSRS and make it more convenient to use.

It's an experiment project and not ready for production. The notebook is just a demo to test the idea. And it's still work in progress.

## Notes

1. The simplified optimization process assumes the last memory state is constant and accurate. It's not true in reality.
2. I don't know how to calculate the gradient for weights related to memory difficulty. So I just set them to 0.