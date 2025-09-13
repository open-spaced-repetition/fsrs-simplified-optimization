# Introduction

FSRS is a time-series model for predicting the memory of spaced repetition users. Its originial version requires BPTT (back propagation through time) to train the model. However, most programming languages don't have mature and efficient deep learning libraries that support BPTT. Therefore, we try to simplify the optimization process of FSRS and make it more convenient to use.

It's an experiment project and not ready for production. The notebook is just a demo to test the idea. And it's still work in progress.

## Notes

1. The simplified optimization process assumes the last memory state is constant and accurate. It's not true in reality.
2. One-step BP cannot calculate the gradient for parameters related to memory difficulty because the last stability doesn't rely on the last difficulty. So I make the last S rely on the last D as a trick and it suprisingly improves the log loss.
3. The benchmark result of One-step BP is recorded here: [Expt/one step bp by L-M-Sherlock · Pull Request #261 · open-spaced-repetition/srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark/pull/261)
