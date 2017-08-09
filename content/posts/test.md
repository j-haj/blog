---
title: "Test"
date: 2017-08-02T06:04:51-05:00
draft: false
---
# Here is a test
An interesting engine that I can't seem to get correct...

```c++
// Some C++ code with a comment...
Eigen::MatrixXd random_matrix(size_t rows, size_t cols) {
  Eigen::MatrixXd m (rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      m(i, j) = 0;
    }
  }
  return m;
}
```
another snippet...
```c++
/**
 * Returns a random value sample from a normal distribution with mean `mean` and
 * standard deviation `std`.
 *
 * @param mean mean of the normal distribution
 * @param std standard deviation of the normal distribution
 *
 * @return random sample from N(mean, std)
 */
double random_double(const double mean, const double std) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<double> d(mean, std);
  return d(gen);
}
```
and an attempt at some Rust code...

```
#[stable(feature = "binary_heap_peek_mut", since = "1.12.0")]
impl<'a, T: Ord> Drop for PeekMut<'a, T> {
  pub fn drop(&mut self) {
    if self.sift {
      self.heap.sift_down(0);
    }
  }
}
```

lastly, some Go and Python code. First Go:

```go
// HexToBase64 takes an input string that is a hex representation and returns
// its base64 representation
func HexToBase64(s string) (string, error) {
	const base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwx" +
		"yz0123456789+/"
	length := utf8.RuneCountInString(s)

	// Pad input
	inputPad, padLength := base64Pad(s)

	bytes, err := hex.DecodeString(s + inputPad)
	if err != nil {
		return "", err
	}

	numBytes := len(bytes)
	result := make([]rune, (length+padLength)*4/6)
	byteIndex := len(result) - 1
	for i := numBytes; i > 0; i -= 3 {
		byteTriple := bytes[i-3 : i]
		val := (2<<15)*int(byteTriple[0]) + (2<<7)*int(byteTriple[1]) +
			int(byteTriple[2])
		for j := 0; j < 4; j++ {
			c := (val >> uint(6*j)) & 0x3F
			result[byteIndex-j] = rune(base64_chars[c])
		}
		byteIndex -= 4
	}
	return string(result), nil
}
```

and now some Python!

```python
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self,
                 loss_func=None,
                 approx_func=None,
                 gradient=None,
                 learning_rate=0.0001):
        super(SGD, self).__init__(loss_func, approx_func, gradient, learning_rate)

    def update_weights(self, data):
        """Runs a single weight update and returns the updated weights"""
        # Need to make sure a gradient and func functions are passed in

        x_vals = data[:, :-1]
        y = data[:, -1]
        aggregate_grad = 0

        for i in range(y.size):
            # Get gradient
            aggregate_grad = np.add(aggregate_grad,
                    self.loss_func.gradient(x_vals[i, :], y[i]))

        aggregate_grad = np.divide(aggregate_grad, y.size)
        logger.debug("SGD gradient: {}".format(aggregate_grad))
        self.approx_func.update_parameters(self._learning_rate * aggregate_grad)
return 1
```
