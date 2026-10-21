use std::path::Path;

#[test]
fn onnx_scores_a_demo() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let demo = root
        .join("data/raw/pro_demos")
        .join("demos_shard-europe-020c2c03_2393178_mouz-nxt-vs-ursa-m2-mirage.dem");
    let model = root.join("artifacts/valveguard_cnn_v1_onnx/model.onnx");
    if !demo.exists() || !model.exists() {
        return;
    }
    let result = valveguard_core::app::score_demo(&demo, &model).unwrap();
    assert_eq!(result["summary"]["players"], 10);
    assert!(result["players"].as_array().unwrap().len() == 10);
}
