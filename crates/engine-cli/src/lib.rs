//! `corbul` — command-line front end for the companion engine.
//!
//! * [`template`] – a small, safe Jinja subset for chat templates, plus built-in fallbacks.
//! * [`http`]     – a strict, bounded HTTP/1.1 request reader and response writers.
//! * [`load`]     – loads a model folder or GGUF file together with its tokenizer and template.
//! * [`serve`]    – OpenAI-compatible and Ollama-style local server.
/// Write a line to standard output. A closed pipe (`corbul capabilities | head`) ends the program quietly
/// instead of panicking like `println!`; any other write error is reported once and exits with 1.
#[macro_export]
macro_rules! outln {
    () => { $crate::out_str("\n") };
    ($($arg:tt)*) => { $crate::out_str(&format!("{}\n", format_args!($($arg)*))) };
}
/// Like [`outln!`] without the newline.
#[macro_export]
macro_rules! out {
    ($($arg:tt)*) => { $crate::out_str(&format!("{}", format_args!($($arg)*))) };
}

/// Standard output with the closed-pipe rule of [`outln!`]. In unit tests the text goes through the test
/// harness' capture instead.
pub fn out_str(s: &str) {
    #[cfg(test)]
    {
        print!("{s}");
    }
    #[cfg(not(test))]
    {
        use std::io::Write;
        let mut o = std::io::stdout().lock();
        if let Err(e) = o.write_all(s.as_bytes()).and_then(|()| o.flush()) {
            if e.kind() != std::io::ErrorKind::BrokenPipe {
                eprintln!("corbul: cannot write to standard output: {e}");
                std::process::exit(1);
            }
            std::process::exit(0);
        }
    }
}

pub mod anthropic;
pub mod args;
pub mod auth;
pub mod cmds;
pub mod constraint;
pub mod enc_cmd;
pub mod famrun;
pub mod famserve;
pub mod fast;
pub mod genloop;
pub mod gpu;
pub mod http;
pub mod kvfit_cmd;
pub mod kvmap_cmd;
pub mod load;
pub mod lorasrc;
pub mod merge_cmd;
pub mod metrics;
pub mod modelfile;
pub mod packs_cmd;
pub mod quant_cmd;
pub mod quotes;
pub mod rag_cmd;
pub mod registry;
pub mod runner;
pub mod scan_cmd;
pub mod search_cmd;
pub mod serve;
pub mod shutdown;
pub mod steer_cmd;
pub mod template;
pub mod tools;
pub mod webui;
pub mod wm;
