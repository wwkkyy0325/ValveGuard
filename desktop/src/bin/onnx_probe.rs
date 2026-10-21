use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: onnx_probe <demo>");
        std::process::exit(2);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let model = root.join("artifacts/valveguard_cnn_v1_onnx/model.onnx");
    match valveguard_core::app::score_demo(Path::new(&args[1]), &model) {
        Ok(value) => {
            println!(
                "players={} flagged={}",
                value["summary"]["players"], value["summary"]["flagged"]
            );
            for player in value["players"].as_array().unwrap() {
                println!(
                    "{} {} {} {}",
                    player["rank"], player["steamid"], player["score"], player["kill_count"]
                );
            }
        }
        Err(error) => {
            eprintln!("{error:#}");
            std::process::exit(1);
        }
    }
}
