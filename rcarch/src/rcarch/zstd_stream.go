// This program is a helper that can handle zstd compression and decompression
// on systems that don't have the zstandard python package with a recent enough
// version, e.g. mac os 15.5.
package main

import (
	"errors"
	"flag"
	"io"
	"log"
	"net"
	"os"

	"github.com/klauspost/compress/zstd"
)

type switchableReadWriter struct{ wrapped io.ReadWriter }

func (s *switchableReadWriter) Read(p []byte) (int, error) { return s.wrapped.Read(p) }

func (s *switchableReadWriter) Write(p []byte) (int, error) { return s.wrapped.Write(p) }

func newDecoder(r io.Reader) *zstd.Decoder {
	decoder, err := zstd.NewReader(r, zstd.WithDecoderConcurrency(1) /* TODO needed? */)
	if err != nil {
		log.Fatalf("creating zstd decoder failed: %v", err)
	}
	return decoder
}

func newEncoder(w io.Writer) *zstd.Encoder {
	encoder, err := zstd.NewWriter(w,
		zstd.WithEncoderLevel(zstd.SpeedDefault),
		zstd.WithEncoderCRC(false),
		zstd.WithZeroFrames(false),
	)
	if err != nil {
		log.Fatalf("creating zstd encoder failed: %v", err)
	}
	return encoder
}

type readResult struct {
	n   int
	err error
}

type decoderWrapper struct {
	inputWrapper *inputWrapperForDecoder
	decoder      *zstd.Decoder
	decoderReads chan (readResult)
}

func newDecoderWrapper() *decoderWrapper {
	dw := decoderWrapper{inputWrapper: newInputWrapperForDecoder()}
	dw.decoder = newDecoder(dw.inputWrapper)
	return &dw
}

func (dw *decoderWrapper) Read(buf []byte) (int, error) {
	if dw.decoderReads == nil { // there's no decoder Read in flight
		dw.decoderReads = make(chan readResult, 1)
		go func(decoderReads chan readResult) {
			n, err := dw.decoder.Read(buf)
			decoderReads <- readResult{n: n, err: err}
		}(dw.decoderReads)
	}
	select {
	case result := <-dw.decoderReads:
		dw.decoderReads = nil
		return result.n, result.err
	case <-dw.inputWrapper.newInputNeeded: // decoder called inputWrapper.Read but the input was already at eof
		return 0, io.EOF
	}
}

func (dw *decoderWrapper) setInput(input io.Reader) {
	dw.inputWrapper.newInputNeeded = make(chan struct{})
	dw.inputWrapper.inputs <- input
}

type inputWrapperForDecoder struct {
	input          io.Reader
	inputs         chan io.Reader
	newInputNeeded chan struct{} // closed when Read is called but we're at eof
	signalEOF      bool
}

func newInputWrapperForDecoder() *inputWrapperForDecoder {
	return &inputWrapperForDecoder{inputs: make(chan io.Reader, 1)}
}

func (iwfd *inputWrapperForDecoder) Read(buf []byte) (int, error) {
	for {
		if iwfd.input == nil {
			if iwfd.signalEOF {
				close(iwfd.newInputNeeded)
				iwfd.signalEOF = false
			}
			iwfd.input = <-iwfd.inputs
		}
		n, err := iwfd.input.Read(buf)
		if iwfd.signalEOF = errors.Is(err, io.EOF); iwfd.signalEOF {
			iwfd.input, err = nil, nil
		}
		if n > 0 || err != nil {
			return n, err
		}
	}
}

func accept(listener net.Listener, stopChan <-chan struct{}) net.Conn {
	acceptChan := make(chan net.Conn)
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				continue
			}
			acceptChan <- conn
			return
		}
	}()
	for {
		select {
		case conn := <-acceptChan:
			return conn
		case <-stopChan:
			return nil
		}
	}
}

func stream(listener net.Listener, stopChan <-chan struct{}, decompress bool) {
	var decoderWrapper *decoderWrapper
	var encoder *zstd.Encoder
	var reader io.Reader
	var writer io.Writer
	var srw switchableReadWriter

	if echo := os.Getenv("ZSTD_STREAM_ECHO"); echo != "" && echo != "0" {
		reader, writer = &srw, &srw
	} else {
		if decompress {
			decoderWrapper = newDecoderWrapper()
			defer decoderWrapper.decoder.Close()
			reader, writer = decoderWrapper, &srw
		} else {
			encoder = newEncoder(&srw)
			defer encoder.Close()
			reader, writer = &srw, encoder
		}
	}

	buf := make([]byte, 64*1024)
	for {
		conn := accept(listener, stopChan)
		if conn == nil {
			break
		}
		if decoderWrapper != nil {
			decoderWrapper.setInput(conn)
		}
		srw.wrapped = conn
		var err error
		// it's important to keep reusing the same buffer, because when
		// the decoder issues a Read and we leave it hanging until we
		// get to the next connection, the Read still references the
		// buffer used by the previous CopyBuffer, so it wouldn't work
		// with just io.Copy
		if _, err = io.CopyBuffer(writer, reader, buf); err == nil && encoder != nil {
			err = encoder.Flush()
		}
		if err != nil {
			log.Fatalln("error while streaming:", err)
		}
		conn.Close()
	}
}

func main() {
	decompress := flag.Bool("decompress", false, "decompress input, instead of compressing")
	flag.Parse()

	// stdout is a bound and listen()ing socket
	listener, err := net.FileListener(os.NewFile(uintptr(1), "listener"))
	if err != nil {
		log.Fatalf("failed to open listener from fd 1: %v", err)
	}
	defer listener.Close()

	exitChan := make(chan struct{})
	go func() {
		// stdin is a pipe; when Read succeeds, it means the parent
		// process wants us to exit or is no longer running
		os.Stdin.Read(make([]byte, 1))
		exitChan <- struct{}{}
	}()

	stream(listener, exitChan, *decompress)
}
