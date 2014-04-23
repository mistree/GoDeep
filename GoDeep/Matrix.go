// Matrix
package GoDeep

func NewMatrix(Row int, Col int) [][]float64 {
	Base := make([]float64, Row*Col)
	Outer := make([][]float64, Row)
	for i := range Outer {
		Outer[i] = Base[i*Col : (i+1)*Col]
	}
	return Outer
}
