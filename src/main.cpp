/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [femelo](https://github.com/femelo)
 * @date    2023
 * @brief   Python bindings for [qwen3-asr.cpp](https://github.com/predict-woo/qwen3-asr.cpp) using Pybind11
 *
 * @par
 * COPYRIGHT NOTICE: (c) 2026.  All rights reserved.
 ********************************************************************************
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "qwen3_asr.h"
#include "forced_aligner.h"


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define DEF_RELEASE_GIL(name, fn, doc) \
    m.def(name, fn, doc, py::call_guard<py::gil_scoped_release>())


namespace py = pybind11;
using namespace qwen3_asr;

PYBIND11_MODULE(_py_qwen3_asr_cpp, m) {
    m.doc() = "Python bindings for Qwen3 ASR engine";

    // 1. Bind the transcribe_params struct
    py::class_<transcribe_params>(m, "TranscribeParams")
        .def(py::init<>())
        .def_readwrite("max_tokens", &transcribe_params::max_tokens)
        .def_readwrite("language", &transcribe_params::language)
        .def_readwrite("n_threads", &transcribe_params::n_threads)
        .def_readwrite("print_progress", &transcribe_params::print_progress)
        .def_readwrite("print_timing", &transcribe_params::print_timing);

    // 2. Bind the transcribe_result struct
    py::class_<transcribe_result>(m, "TranscribeResult")
        .def_readonly("language", &transcribe_result::language)
        .def_readonly("text", &transcribe_result::text)
        .def_readonly("tokens", &transcribe_result::tokens)
        .def_readonly("success", &transcribe_result::success)
        .def_readonly("error_msg", &transcribe_result::error_msg)
        .def_readonly("t_load_ms", &transcribe_result::t_load_ms)
        .def_readonly("t_mel_ms", &transcribe_result::t_mel_ms)
        .def_readonly("t_encode_ms", &transcribe_result::t_encode_ms)
        .def_readonly("t_decode_ms", &transcribe_result::t_decode_ms)
        .def_readonly("t_total_ms", &transcribe_result::t_total_ms);

    // 3. Bind the main Qwen3ASR class
    py::class_<Qwen3ASR>(m, "Qwen3ASR")
        .def(py::init<>())
        .def("load_model", &Qwen3ASR::load_model, py::arg("model_path"))
        
        // Wrap the file-path transcription
        .def("transcribe_file", 
             static_cast<transcribe_result (Qwen3ASR::*)(const std::string&, const transcribe_params&)>(&Qwen3ASR::transcribe),
             py::arg("audio_path"), py::arg("params") = transcribe_params())

        // Wrap the raw buffer transcription using NumPy for Python efficiency
        .def("transcribe_samples", [](Qwen3ASR &self, py::array_t<float> samples, const transcribe_params &params) {
            py::buffer_info buf = samples.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Audio samples must be a 1D array");
            }
            return self.transcribe(static_cast<float*>(buf.ptr), static_cast<int>(buf.size), params);
        }, py::arg("samples"), py::arg("params") = transcribe_params())

        .def("set_progress_callback", &Qwen3ASR::set_progress_callback)
        .def("get_error", &Qwen3ASR::get_error)
        .def("is_loaded", &Qwen3ASR::is_loaded);

    m.def("load_audio_file", [](const std::string &path) {
        std::vector<float> samples;
        int sample_rate = 0;
        
        bool success = load_audio_file(path, samples, sample_rate);
        
        if (!success) {
            throw std::runtime_error("Failed to load audio file: " + path);
        }

        // Create the NumPy array from vector data
        py::array_t<float> audio_array(static_cast<py::ssize_t>(samples.size()), samples.data());

        // Return as a tuple: (numpy_array, sample_rate)
        return py::make_tuple(audio_array, sample_rate);
    }, "Loads a WAV file and returns (audio_array, sample_rate)");

    // 1. Bind Aligned Word and Result
    py::class_<aligned_word>(m, "AlignedWord")
        .def_readonly("word", &aligned_word::word)
        .def_readonly("start", &aligned_word::start)
        .def_readonly("end", &aligned_word::end);

    py::class_<alignment_result>(m, "AlignmentResult")
        .def_readonly("words", &alignment_result::words)
        .def_readonly("success", &alignment_result::success)
        .def_readonly("error_msg", &alignment_result::error_msg)
        .def_readonly("t_total_ms", &alignment_result::t_total_ms);

    // 2. Bind Hyperparameters (Optional but useful for debugging)
    py::class_<forced_aligner_hparams>(m, "ForcedAlignerHParams")
        .def_readonly("vocab_size", &forced_aligner_hparams::vocab_size)
        .def_readonly("text_hidden_size", &forced_aligner_hparams::text_hidden_size);

    // 3. Bind the ForcedAligner Class
    py::class_<ForcedAligner>(m, "ForcedAligner")
        .def(py::init<>())
        .def("load_model", &ForcedAligner::load_model, py::arg("model_path"))
        .def("load_korean_dict", &ForcedAligner::load_korean_dict, py::arg("dict_path"))
        
        // Wrap file-based alignment
        .def("align_file", 
             static_cast<alignment_result (ForcedAligner::*)(const std::string&, const std::string&, const std::string&)>(&ForcedAligner::align),
             py::arg("audio_path"), py::arg("text"), py::arg("language") = "")

        // Wrap raw buffer alignment (NumPy support)
        .def("align_samples", [](ForcedAligner &self, py::array_t<float> samples, const std::string &text, const std::string &language) {
            py::buffer_info buf = samples.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Audio samples must be a 1D array");
            }
            return self.align(static_cast<float*>(buf.ptr), static_cast<int>(buf.size), text, language);
        }, py::arg("samples"), py::arg("text"), py::arg("language") = "")

        .def("get_error", &ForcedAligner::get_error)
        .def("is_loaded", &ForcedAligner::is_loaded)
        .def("get_hparams", &ForcedAligner::get_hparams);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
