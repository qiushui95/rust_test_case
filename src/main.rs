use std::thread;
use rust_test_case::assets::Assets;
use rust_test_case::auto_gui::{AutoGui, FindImageRegion, FindImageResultFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    thread::sleep(std::time::Duration::from_secs(1));

    // 开启 debug 方便观察匹配过程日志
    let mut gui = AutoGui::new(false)?;

    // 测试函数：传入原图、缩放图，自动读取缩放图宽度作为 template_width
    fn test(
        gui: &mut AutoGui,
        asset_name: &str,
        template_width: Option<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "----------{} start {template_width:?}------------",
            asset_name
        );

        let start_time = std::time::Instant::now();

        let results = gui.find_image_on_screen(
            asset_name,
            0.9,
            None,
            template_width,
            None,
        )?;

        let duration = start_time.elapsed();

        println!(
            "width: {}, height: {}, list: {}",
            results.width,
            results.height,
            results.list.len()
        );

        for result in results.list {
            println!("({},{}),{}", result.left, result.top, result.precision);
        }

        println!(
            "++++++++++{} end {template_width:?} {}ms ++++++++++++",
            asset_name,
            duration.as_millis()
        );

        Ok(())
    }

    test(&mut gui, "aa.png", None)?;
    test(&mut gui, "aa.png", Some(130))?;
    test(&mut gui, "aa.png", Some(110))?;
    test(&mut gui, "aa.png", Some(80))?;
    test(&mut gui, "aa.png", Some(50))?;
    test(&mut gui, "aa.png", Some(50))?;
    test(&mut gui, "aa.png", Some(90))?;
    Ok(())
}
