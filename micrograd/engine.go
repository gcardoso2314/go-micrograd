package micrograd

import (
	"fmt"
	"math"
)

type Value struct {
	Data     float64
	Grad     float64
	Children []*Value
	backward func()
	op       string
}

func NewValue(data float64, children []*Value, op string) *Value {
	return &Value{data, 0, children, func() {}, op}
}

func NewLeafValue(data float64) *Value {
	return NewValue(data, []*Value{}, "")
}

func MatrixToValues(x [][]float64) [][]*Value {
	out := make([][]*Value, len(x))
	for i, row := range x {
		out[i] = make([]*Value, len(row))
		for j, val := range row {
			out[i][j] = NewLeafValue(val)
		}
	}

	return out
}

func (v *Value) Print() {
	fmt.Printf("Value(data=%.4f, grad=%.4f)\n", v.Data, v.Grad)
}

func (v *Value) Add(other interface{}) *Value {
	var otherValue *Value
	switch t := other.(type) {
	case *Value:
		otherValue = t
	case float64:
		otherValue = NewLeafValue(t)
	case int:
		otherValue = NewLeafValue(float64(t))
	}

	out := NewValue(v.Data+otherValue.Data, []*Value{v, otherValue}, "+")
	var backward = func() {
		v.Grad += out.Grad
		otherValue.Grad += out.Grad
	}
	out.backward = backward
	return out
}

func (v *Value) Neg() *Value {
	return v.Mul(-1.0)
}

func (v *Value) Mul(other interface{}) *Value {
	var otherValue *Value
	switch t := other.(type) {
	case *Value:
		otherValue = t
	case float64:
		otherValue = NewLeafValue(t)
	case int:
		otherValue = NewLeafValue(float64(t))
	}

	out := NewValue(v.Data*otherValue.Data, []*Value{v, otherValue}, "*")
	var backward = func() {
		v.Grad += out.Grad * otherValue.Data
		otherValue.Grad += out.Grad * v.Data
	}
	out.backward = backward
	return out
}

func (v *Value) Pow(other float64) *Value {
	out := NewValue(math.Pow(v.Data, other), []*Value{v}, fmt.Sprintf("**%.0f", other))
	var backward = func() {
		v.Grad += out.Grad * (other * math.Pow(v.Data, other-1.0))
	}
	out.backward = backward
	return out
}

func (v *Value) Log() *Value {
	// clamp the values to avoid log(0)
	out := NewValue(math.Max(math.Log(v.Data), -100.0), []*Value{v}, "log")
	var backward = func() {
		v.Grad += out.Grad / (v.Data + 1e-4) // avoid div by zero
	}
	out.backward = backward
	return out
}

func (v *Value) ReLU() *Value {
	var outData float64
	if v.Data <= 0.0 {
		outData = 0.0
	} else {
		outData = v.Data
	}
	out := NewValue(outData, []*Value{v}, "ReLU")
	var backward = func() {
		if v.Data > 0 {
			v.Grad += out.Grad
		}
	}
	out.backward = backward
	return out
}

func (v *Value) Tanh() *Value {
	outData := math.Tanh(v.Data)
	out := NewValue(outData, []*Value{v}, "Tanh")
	var backward = func() {
		v.Grad += (1 - math.Pow(outData, 2.0)) * out.Grad
	}
	out.backward = backward
	return out
}

func (v *Value) Sigmoid() *Value {
	outData := 1.0 / (1.0 + math.Exp(-v.Data))
	out := NewValue(outData, []*Value{v}, "Sigmoid")
	var backward = func() {
		v.Grad += out.Grad * outData * (1 - outData)
	}
	out.backward = backward
	return out
}

func (v *Value) Backward() {
	v.Grad = 1.0

	// Create a topological order of nodes
	visited := map[*Value]bool{}
	topo := []*Value{}
	var buildTopo func(node *Value)
	buildTopo = func(node *Value) {
		if visited[node] {
			return
		}

		visited[node] = true
		for _, c := range node.Children {
			buildTopo(c)
		}
		topo = append(topo, node)
	}
	buildTopo(v)

	// Call backward pass on each node in reverse topological order
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}
