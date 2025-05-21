package main

import (
	"bytes"
	"io"
	"math/rand"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func randPrintable(t *testing.T, size int) []byte {
	const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}"
	buf := make([]byte, size)
	r := rand.New(rand.NewSource(1234))
	for i := range buf {
		buf[i] = chars[r.Intn(len(chars))]
	}
	return buf
}

func runStream(t *testing.T, decompress bool) (string, chan struct{}, chan struct{}) {
	listener, err := net.Listen("tcp", "localhost:0")
	require.NoError(t, err)
	stopChan, stoppedChan := make(chan struct{}), make(chan struct{})
	go func() {
		stream(listener, stopChan, decompress)
		close(stoppedChan)
	}()
	return listener.Addr().String(), stopChan, stoppedChan
}

func process(t *testing.T, addr string, input []byte) []byte {
	conn, err := net.Dial("tcp", addr)
	require.NoError(t, err)
	go func() {
		_, err := io.Copy(conn, bytes.NewReader(input))
		require.NoError(t, err)
		require.NoError(t, conn.(*net.TCPConn).CloseWrite())
	}()
	var output bytes.Buffer
	_, err = io.Copy(&output, conn)
	require.NoError(t, err)
	return output.Bytes()
}

func requireClosed(t *testing.T, c chan struct{}) {
	select {
	case <-c:
		return
	case <-time.After(3 * time.Second):
		require.Fail(t, "channel isn't closed, stream() must still be running")
	}
}

func testBlobs(t *testing.T, nblobs int, sizes []int) {
	caddr, cstop, cstopped := runStream(t, false)
	daddr, dstop, dstopped := runStream(t, true)
	for i := 0; i < nblobs; i++ {
		in := randPrintable(t, sizes[i%len(sizes)])
		compressed := process(t, caddr, in)
		out := process(t, daddr, compressed)
		require.Equal(t, len(in), len(out))
		require.True(t, bytes.Equal(in, out))
	}
	close(cstop)
	close(dstop)
	requireClosed(t, cstopped)
	requireClosed(t, dstopped)
}

func TestMultipleShortBlobs(t *testing.T) {
	t.Parallel()
	testBlobs(t, 10, []int{100})
}

func TestMultipleLongBlobs(t *testing.T) {
	t.Parallel()
	testBlobs(t, 10, []int{300 * 1000})
}

func TestOneShortBlob(t *testing.T) {
	t.Parallel()
	testBlobs(t, 1, []int{50})
}

func TestOneLongBlob(t *testing.T) {
	t.Parallel()
	testBlobs(t, 1, []int{600 * 1000})
}

func TestNoBlobs(t *testing.T) {
	t.Parallel()
	testBlobs(t, 0, []int{100})
}

func TestOneZeroSizeBlob(t *testing.T) {
	t.Parallel()
	testBlobs(t, 1, []int{0})
}

func TestSeveralZeroSizeBlobs(t *testing.T) {
	t.Parallel()
	testBlobs(t, 10, []int{0})
}

func TestFunkySizeBlobs(t *testing.T) {
	t.Parallel()
	testBlobs(t, 10, []int{10, 0, 300 * 1000, 1000, 100 * 1000})
}
