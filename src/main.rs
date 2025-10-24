use rust_test_case::assets::Assets;
use rust_test_case::matcher::ImageMatcher;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let template_file = Assets::get("aa.png").unwrap();
    let screen_file = Assets::get("screen.png").unwrap();

    let template_image = image::load_from_memory(&template_file.data).unwrap();
    let screen_image = image::load_from_memory(&screen_file.data).unwrap();

    let matcher = ImageMatcher::new(template_image,true,None)?;

    let results = matcher.start_matching(screen_image, 0.9, None, None)?;

    for result in results.list {
        println!("left: {}, top: {}, threshold: {}", result.left, result.top, result.precision);
    }
    
    Ok(())
}
