package micrograd

type LossFunction func(ytrue, ypred []*Value) *Value

func MSE(ytrue, ypred []*Value) *Value {
	loss := NewLeafValue(0)
	for i := 0; i < len(ytrue); i++ {
		loss = loss.Add(ytrue[i].Add(ypred[i].Neg()).Pow(2))
	}
	return loss.Mul(1.0 / float64(len(ytrue)))
}

func HingeLoss(ytrue, ypred []*Value) *Value {
	loss := NewLeafValue(0)
	for i := 0; i < len(ypred); i++ {
		sampleLoss := ytrue[i].Neg().Mul(ypred[i]).Add(1).ReLU()
		loss = loss.Add(sampleLoss)
	}
	return loss.Mul(1.0 / float64(len(ypred)))
}

func BinaryCrossEntropy(ytrue, ypred []*Value) *Value {
	loss := NewLeafValue(0)
	for i := 0; i < len(ytrue); i++ {
		sampleLoss := ytrue[i].Mul(ypred[i].Log()).Add(ytrue[i].Neg().Add(1).Mul(ypred[i].Neg().Add(1).Log()))
		loss = loss.Add(sampleLoss)
	}
	return loss.Mul(-1.0 / float64(len(ytrue)))
}
