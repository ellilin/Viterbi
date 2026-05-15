#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class ConvolutionalCode {
public:
    using Polynomial = std::uint32_t;

    ConvolutionalCode(int input_bits,
                      int output_bits,
                      int memory,
                      std::vector<std::vector<Polynomial>> generators)
        : k_(input_bits),
          n_(output_bits),
          memory_(memory),
          generators_(std::move(generators)) {
        if (k_ <= 0 || n_ <= 0 || memory_ < 0) {
            throw std::invalid_argument("Invalid convolutional code dimensions");
        }
        if (memory_ >= 31) {
            throw std::invalid_argument("Memory must be below 31 for uint32 masks");
        }
        if (static_cast<int>(generators_.size()) != n_) {
            throw std::invalid_argument("Generator matrix must have n rows");
        }
        const Polynomial allowed_mask =
            memory_ == 31 ? std::numeric_limits<Polynomial>::max()
                          : ((Polynomial{1} << (memory_ + 1)) - 1);
        for (const auto& row : generators_) {
            if (static_cast<int>(row.size()) != k_) {
                throw std::invalid_argument("Generator matrix must have k columns");
            }
            for (Polynomial g : row) {
                if ((g & ~allowed_mask) != 0) {
                    throw std::invalid_argument("Generator polynomial exceeds memory");
                }
            }
        }
    }

    int inputBits() const { return k_; }
    int outputBits() const { return n_; }
    int memory() const { return memory_; }

    int stateCount() const {
        if (k_ * memory_ >= 30) {
            throw std::overflow_error("Too many trellis states for this demo");
        }
        return 1 << (k_ * memory_);
    }

    int inputSymbolCount() const {
        if (k_ >= 30) {
            throw std::overflow_error("Too many input symbols for this demo");
        }
        return 1 << k_;
    }

    int tailSymbols() const { return memory_; }

    int nextState(int state, int input_symbol) const {
        if (memory_ == 0) {
            return 0;
        }

        int next = 0;
        for (int input = 0; input < k_; ++input) {
            const int current_bit = (input_symbol >> input) & 1;
            const int stream_state = (state >> (input * memory_)) & ((1 << memory_) - 1);
            const int next_stream_state =
                current_bit | ((stream_state << 1) & ((1 << memory_) - 1));
            next |= next_stream_state << (input * memory_);
        }
        return next;
    }

    int outputSymbol(int state, int input_symbol) const {
        int output = 0;
        for (int out = 0; out < n_; ++out) {
            int parity = 0;
            for (int input = 0; input < k_; ++input) {
                const Polynomial generator = generators_[out][input];
                if (generator & 1U) {
                    parity ^= (input_symbol >> input) & 1;
                }

                const int stream_state =
                    memory_ == 0 ? 0 : (state >> (input * memory_)) & ((1 << memory_) - 1);
                for (int delay = 1; delay <= memory_; ++delay) {
                    if ((generator >> delay) & 1U) {
                        parity ^= (stream_state >> (delay - 1)) & 1;
                    }
                }
            }
            output |= parity << out;
        }
        return output;
    }

private:
    int k_;
    int n_;
    int memory_;
    std::vector<std::vector<Polynomial>> generators_;
};

class ConvolutionalEncoder {
public:
    explicit ConvolutionalEncoder(const ConvolutionalCode& code) : code_(code) {}

    std::vector<int> encode(const std::vector<int>& input_bits, bool terminate = true) const {
        if (input_bits.size() % code_.inputBits() != 0) {
            throw std::invalid_argument("Input length must be divisible by k");
        }

        std::vector<int> encoded;
        encoded.reserve((input_bits.size() / code_.inputBits() + code_.tailSymbols()) *
                        code_.outputBits());

        int state = 0;
        for (std::size_t pos = 0; pos < input_bits.size(); pos += code_.inputBits()) {
            const int input_symbol = packBits(input_bits, pos, code_.inputBits());
            appendOutput(encoded, code_.outputSymbol(state, input_symbol));
            state = code_.nextState(state, input_symbol);
        }

        if (terminate) {
            for (int i = 0; i < code_.tailSymbols(); ++i) {
                appendOutput(encoded, code_.outputSymbol(state, 0));
                state = code_.nextState(state, 0);
            }
        }

        return encoded;
    }

private:
    static int packBits(const std::vector<int>& bits, std::size_t offset, int count) {
        int symbol = 0;
        for (int i = 0; i < count; ++i) {
            if (bits[offset + i] != 0 && bits[offset + i] != 1) {
                throw std::invalid_argument("Bits must be 0 or 1");
            }
            symbol |= bits[offset + i] << i;
        }
        return symbol;
    }

    void appendOutput(std::vector<int>& encoded, int output_symbol) const {
        for (int out = 0; out < code_.outputBits(); ++out) {
            encoded.push_back((output_symbol >> out) & 1);
        }
    }

    const ConvolutionalCode& code_;
};

class ViterbiDecoder {
public:
    explicit ViterbiDecoder(const ConvolutionalCode& code) : code_(code) {}

    std::vector<int> decode(const std::vector<int>& received_bits,
                            bool terminated_to_zero = true) const {
        if (received_bits.size() % code_.outputBits() != 0) {
            throw std::invalid_argument("Received length must be divisible by n");
        }

        const int steps = static_cast<int>(received_bits.size() / code_.outputBits());
        const int states = code_.stateCount();
        const int inputs = code_.inputSymbolCount();
        const int inf = std::numeric_limits<int>::max() / 4;

        std::vector<int> metric(states, inf);
        std::vector<int> next_metric(states, inf);
        std::vector<std::vector<int>> prev_state(steps, std::vector<int>(states, -1));
        std::vector<std::vector<int>> prev_input(steps, std::vector<int>(states, -1));

        metric[0] = 0;
        for (int step = 0; step < steps; ++step) {
            std::fill(next_metric.begin(), next_metric.end(), inf);

            for (int state = 0; state < states; ++state) {
                if (metric[state] == inf) {
                    continue;
                }
                for (int input_symbol = 0; input_symbol < inputs; ++input_symbol) {
                    const int next_state = code_.nextState(state, input_symbol);
                    const int expected = code_.outputSymbol(state, input_symbol);
                    const int distance = hammingDistance(received_bits, step, expected);
                    const int candidate = metric[state] + distance;
                    if (candidate < next_metric[next_state]) {
                        next_metric[next_state] = candidate;
                        prev_state[step][next_state] = state;
                        prev_input[step][next_state] = input_symbol;
                    }
                }
            }
            metric.swap(next_metric);
        }

        int state = terminated_to_zero
                        ? 0
                        : static_cast<int>(std::min_element(metric.begin(), metric.end()) -
                                           metric.begin());

        std::vector<int> decoded_symbols(steps);
        for (int step = steps - 1; step >= 0; --step) {
            const int input_symbol = prev_input[step][state];
            if (input_symbol < 0) {
                throw std::runtime_error("Viterbi traceback failed");
            }
            decoded_symbols[step] = input_symbol;
            state = prev_state[step][state];
        }

        std::vector<int> decoded_bits;
        decoded_bits.reserve(static_cast<std::size_t>(steps) * code_.inputBits());
        for (int symbol : decoded_symbols) {
            for (int input = 0; input < code_.inputBits(); ++input) {
                decoded_bits.push_back((symbol >> input) & 1);
            }
        }

        if (terminated_to_zero) {
            const int tail_bits = code_.tailSymbols() * code_.inputBits();
            decoded_bits.resize(decoded_bits.size() - tail_bits);
        }

        return decoded_bits;
    }

private:
    int hammingDistance(const std::vector<int>& received_bits, int step, int expected) const {
        int distance = 0;
        const int offset = step * code_.outputBits();
        for (int out = 0; out < code_.outputBits(); ++out) {
            distance += received_bits[offset + out] != ((expected >> out) & 1);
        }
        return distance;
    }

    const ConvolutionalCode& code_;
};

class BinarySymmetricChannel {
public:
    explicit BinarySymmetricChannel(std::uint32_t seed = std::random_device{}())
        : rng_(seed) {}

    std::vector<int> transmit(const std::vector<int>& bits, double error_probability) {
        if (error_probability < 0.0 || error_probability > 1.0) {
            throw std::invalid_argument("Error probability must be in [0, 1]");
        }

        std::bernoulli_distribution flip(error_probability);
        std::vector<int> received = bits;
        for (int& bit : received) {
            if (flip(rng_)) {
                bit ^= 1;
            }
        }
        return received;
    }

private:
    std::mt19937 rng_;
};

struct SimulationPoint {
    double channel_error_probability = 0.0;
    double decoded_bit_error_rate = 0.0;
    double raw_channel_bit_error_rate = 0.0;
    std::size_t bits = 0;
};

std::vector<int> randomBits(std::size_t count, std::mt19937& rng) {
    std::bernoulli_distribution bit(0.5);
    std::vector<int> bits(count);
    for (int& value : bits) {
        value = bit(rng) ? 1 : 0;
    }
    return bits;
}

std::size_t countErrors(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }
    std::size_t errors = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        errors += lhs[i] != rhs[i];
    }
    return errors;
}

SimulationPoint simulatePoint(const ConvolutionalEncoder& encoder,
                              const ViterbiDecoder& decoder,
                              BinarySymmetricChannel& channel,
                              double channel_error_probability,
                              std::size_t information_bits,
                              int trials,
                              std::mt19937& rng) {
    std::size_t decoded_errors = 0;
    std::size_t raw_errors = 0;
    std::size_t total_bits = 0;
    std::size_t total_channel_bits = 0;

    for (int trial = 0; trial < trials; ++trial) {
        const auto input = randomBits(information_bits, rng);
        const auto encoded = encoder.encode(input);
        const auto received = channel.transmit(encoded, channel_error_probability);
        const auto decoded = decoder.decode(received);

        decoded_errors += countErrors(input, decoded);
        raw_errors += countErrors(encoded, received);
        total_bits += input.size();
        total_channel_bits += encoded.size();
    }

    return {
        channel_error_probability,
        static_cast<double>(decoded_errors) / static_cast<double>(total_bits),
        static_cast<double>(raw_errors) / static_cast<double>(total_channel_bits),
        total_bits,
    };
}

int main() {
    try {
        // NASA/CCSDS-like rate 1/2 code with K = 7, generators 171 and 133 in octal.
        const ConvolutionalCode code(
            1,
            2,
            6,
            {
                {0171},
                {0133},
            });

        const ConvolutionalEncoder encoder(code);
        const ViterbiDecoder decoder(code);
        BinarySymmetricChannel channel(20260515);
        std::mt19937 rng(123456789);

        const std::vector<double> probabilities = {
            0.000, 0.005, 0.010, 0.020, 0.030, 0.040,
            0.050, 0.070, 0.090, 0.110, 0.130, 0.150,
        };

        const std::size_t information_bits = 4096;
        const int trials = 80;

        std::filesystem::create_directories("results");
        std::ofstream csv("results/ber.csv");
        if (!csv) {
            throw std::runtime_error("Cannot open results/ber.csv");
        }
        csv << "channel_error_probability,decoded_bit_error_rate,"
               "raw_channel_bit_error_rate,bits\n";

        std::cout << "p_channel, decoded_BER, raw_channel_BER, bits\n";
        for (double p : probabilities) {
            const auto point =
                simulatePoint(encoder, decoder, channel, p, information_bits, trials, rng);
            csv << std::fixed << std::setprecision(6)
                << point.channel_error_probability << ','
                << std::scientific << std::setprecision(8)
                << point.decoded_bit_error_rate << ','
                << point.raw_channel_bit_error_rate << ','
                << std::defaultfloat << point.bits << '\n';

            std::cout << std::fixed << std::setprecision(3) << point.channel_error_probability
                      << ", " << std::scientific << std::setprecision(4)
                      << point.decoded_bit_error_rate << ", "
                      << point.raw_channel_bit_error_rate << ", "
                      << std::defaultfloat << point.bits << '\n';
        }

        std::cout << "Saved results/ber.csv\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
