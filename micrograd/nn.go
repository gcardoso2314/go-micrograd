package micrograd

import (
	"errors"
	"math/rand"
)

type Neuron struct {
	Weights    []*Value
	Bias       *Value
	activation string
}

type Layer struct {
	Neurons []*Neuron
}

type MLP struct {
	Layers []*Layer
}

func NewNeuron(numInputs int, activation string) *Neuron {
	weights := make([]*Value, numInputs)
	for i := 0; i < numInputs; i++ {
		weights[i] = NewLeafValue(rand.NormFloat64())
	}
	return &Neuron{
		Weights:    weights,
		Bias:       NewLeafValue(0),
		activation: activation,
	}
}

func (n *Neuron) Forward(x []*Value) (*Value, error) {
	// assert len(x) matches number of weights
	if len(x) != len(n.Weights) {
		return nil, errors.New("input data doesn't match expected dimensions")
	}

	act := NewLeafValue(0)
	for i := 0; i < len(n.Weights); i++ {
		act = act.Add(n.Weights[i].Mul(x[i]))
	}
	act = act.Add(n.Bias)
	switch n.activation {
	case "relu":
		act = act.ReLU()
	case "tanh":
		act = act.Tanh()
	case "sigmoid":
		act = act.Sigmoid()
	case "":
		// do nothing
	default:
		return nil, errors.New("unsupported activation function")
	}
	return act, nil
}

func (n *Neuron) Parameters() []*Value {
	params := []*Value{}
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}

func NewLayer(numInputs, numOutputs int, activation string) *Layer {
	neurons := make([]*Neuron, numOutputs)
	for i := 0; i < numOutputs; i++ {
		neurons[i] = NewNeuron(numInputs, activation)
	}
	return &Layer{
		Neurons: neurons,
	}
}

func (l *Layer) Forward(x []*Value) ([]*Value, error) {
	out := make([]*Value, len(l.Neurons))
	for i := 0; i < len(l.Neurons); i++ {
		neuronOut, err := l.Neurons[i].Forward(x)
		if err != nil {
			return nil, err
		}
		out[i] = neuronOut
	}
	return out, nil
}

func (l *Layer) Parameters() []*Value {
	params := []*Value{}
	for _, neuron := range l.Neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

func NewMLP(numInputs int, layerSizes []int) (*MLP, error) {
	if layerSizes[len(layerSizes)-1] != 1 {
		return nil, errors.New("only single neuron ouput MLPs are supported")
	}

	layers := make([]*Layer, len(layerSizes))
	layerDims := append([]int{numInputs}, layerSizes...)
	for i := 0; i < len(layerSizes); i++ {
		if i == len(layerSizes)-1 {
			// Last layer sigmoid for classification
			layers[i] = NewLayer(layerDims[i], layerDims[i+1], "sigmoid")
		} else {
			layers[i] = NewLayer(layerDims[i], layerDims[i+1], "relu")
		}

	}
	return &MLP{
		Layers: layers,
	}, nil
}

func (m *MLP) Forward(x []*Value) (*Value, error) {
	var out []*Value
	out = x
	for _, l := range m.Layers {
		layerOut, err := l.Forward(out)
		if err != nil {
			return nil, err
		}
		out = layerOut
	}

	if len(out) != 1 {
		return nil, errors.New("expected output to only have one value")
	}

	return out[0], nil
}

func (m *MLP) Parameters() []*Value {
	params := []*Value{}
	for _, l := range m.Layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

func (m *MLP) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.Grad = 0
	}
}
