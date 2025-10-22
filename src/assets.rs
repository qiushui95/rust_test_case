use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "icons/"]
pub struct Assets;

impl Assets {
    pub const WX_CONTACT: &'static str = "微信联系人.png";
}
