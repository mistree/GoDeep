// NNTest
package main

import (
	"fmt"
	"github.com/mistree/GoDeep"
	"runtime"
	"time"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	//RBMTest()
	//NNTest()
	DBNTest()
	//SpeedTest()
}

func RBMTest() {
	m := GoDeep.NewRBM(10, 3)
	var Trainer GoDeep.Train
	Trainer.Dropout = 0.1
	Trainer.LearnRate = 1
	Trainer.Momentum = 0.1
	Trainer.Epoch = 20000
	Trainer.ErrorAllowed = 0.1
	start := time.Now()
	Error := m.Train(&[][]float64{[]float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
		[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
		[]float64{1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
		[]float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}}, Trainer)
	fmt.Printf("RBM Finished training in: %s\r\n", time.Since(start))
	fmt.Printf("Error %f\r\n", Error)
	fmt.Println(*m.Feedback(m.Forward(&[]float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1})))
	fmt.Println(*m.Feedback(m.Forward(&[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0})))
	fmt.Println(*m.Feedback(m.Forward(&[]float64{1, 1, 1, 1, 1, 0, 0, 0, 0, 0})))
	fmt.Println(*m.Feedback(m.Forward(&[]float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1})))
}

func NNTest() {
	nn := GoDeep.NewNN([]int{2, 10, 10, 1})
	inputs := [][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
	}

	targets := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}

	var Trainer GoDeep.Train
	Trainer.Dropout = 0
	Trainer.Epoch = 5000
	Trainer.LearnRate = 0.1
	Trainer.Momentum = 0.1
	Trainer.ErrorAllowed = 0.1
	start := time.Now()
	Error, Succeed := nn.Train(&inputs, &targets, Trainer)
	fmt.Printf("NN Finished training in: %s\r\n", time.Since(start))
	if Succeed {
		fmt.Printf("Succeed, Error %f\r\n", Error)
	} else {
		fmt.Printf("Failed, Error %f\r\n", Error)
	}
	fmt.Print(*nn.RoundForward(&inputs[0]))
	fmt.Print(*nn.RoundForward(&inputs[1]))
	fmt.Print(*nn.RoundForward(&inputs[2]))
	fmt.Print(*nn.RoundForward(&inputs[3]))
	fmt.Println("")
}

func DBNTest() {
	d := GoDeep.NewDBN([]int{2, 10, 10, 1})
	var Trainer GoDeep.Train
	Trainer.Dropout = 0.1
	Trainer.LearnRate = 1
	Trainer.Momentum = 0.1
	Trainer.Epoch = 10000
	Trainer.ErrorAllowed = 0.1
	start := time.Now()
	d.Train(&[][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1}}, Trainer)

	///*
	n := d.ToNN()
	Trainer.Epoch = 2000
	Trainer.Dropout = 0.01
	Error, Succeed := n.Train(&[][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1}},
		&[][]float64{
			[]float64{0},
			[]float64{1},
			[]float64{1},
			[]float64{0}}, Trainer)
	fmt.Printf("DBN Finished training in: %s\r\n", time.Since(start))
	if Succeed {
		fmt.Printf("Succeed, Error %f\r\n", Error)
	} else {
		fmt.Printf("Failed, Error %f\r\n", Error)
	}
	fmt.Print(*n.RoundForward(&[]float64{0, 0}))
	fmt.Print(*n.RoundForward(&[]float64{0, 1}))
	fmt.Print(*n.RoundForward(&[]float64{1, 0}))
	fmt.Print(*n.RoundForward(&[]float64{1, 1}))
	fmt.Println("")
	//*/
}

func SpeedTest() {
	start1 := time.Now()
	for i := 0; i < 100; i++ {
		NNTest()
	}
	end1 := time.Now()
	start2 := time.Now()
	for i := 0; i < 100; i++ {
		DBNTest()
	}
	end2 := time.Now()
	fmt.Printf("NN Finished training in: %s\r\n", end1.Sub(start1))
	fmt.Printf("DBN Finished training in: %s\r\n", end2.Sub(start2))
}
