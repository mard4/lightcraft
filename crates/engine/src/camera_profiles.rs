//! Camera colour profiles of our own, pooled from many photos of one camera model.
//!
//! A raw file without colour matrices (Sony ARW, Nikon NEF, Fujifilm RAF) gets its look fitted to its own embedded camera
//! JPEG (`camera_preview`), but one photo shows too little of some colours: a lime shirt covering a
//! few dozen proxy pixels next to a hillside of foliage at the same hue. `lightcraft-cli calibrate`
//! pools the colour pairs of many photos per model and fits one matrix and hue/saturation/value
//! table; photos of that model then only fit their tone and chroma curves.
//!
//! Profiles are JSON files (`<model>.json`) in [`dir`]: `LIGHTCRAFT_CAMERA_PROFILES`, else
//! `<config>/camera-profiles`; a local profile replaces the one built in ([`BUNDLED`], from
//! `assets/camera-profiles/`). They are read once per process; a damaged or hostile file is
//! ignored with a warning. They hold aggregate colour statistics only, never image content.
use lightcraft_color::Mat3;
use lightcraft_pipeline::tone::CameraTone;
use lightcraft_raw::profile::HsvTable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// Format version of the profile files.
const VERSION: u32 = 1;
/// Largest profile file read (a 5 × 72 × 5 table is ~60 KB of JSON).
const MAX_FILE: u64 = 4 << 20;

/// Profiles built into LightCraft (`assets/camera-profiles/`, see `assets/ATTRIBUTION.md`):
/// `(model, JSON)`.
pub const BUNDLED: &[(&str, &str)] = &[
    ("ILCE-7CR", include_str!("../../../assets/camera-profiles/ILCE-7CR.json")),
    ("ILCE-7M4", include_str!("../../../assets/camera-profiles/ILCE-7M4.json")),
    ("X-H2S", include_str!("../../../assets/camera-profiles/X-H2S.json")),
    ("X-T4", include_str!("../../../assets/camera-profiles/X-T4.json")),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraProfile {
    pub version: u32,
    /// The camera model as the files report it (Exif `Model`), e.g. `ILCE-7M4`.
    pub model: String,
    /// Photos and colour pairs the profile was fitted on.
    pub files: usize,
    pub samples: usize,
    /// White-balanced camera RGB (with the baseline exposure) → linear Rec.2020, rows.
    matrix: [[f64; 3]; 3],
    /// Hue/saturation/value correction after `matrix` (linear ProPhoto RGB).
    pub hue_sat: Option<HsvTable>,
    /// For a look profile: the in-camera rendering it was pooled from (`lightcraft_raw::FilmLook::key`, a film
    /// simulation with its tone settings), e.g. `classic-chrome_drp-strong`. `None` for the model's colour profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<String>,
    /// A look profile's tone and chroma curves, for a raw exposed at 100 % dynamic range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tone: Option<CameraTone>,
}

impl CameraProfile {
    pub fn new(model: &str, files: usize, samples: usize, matrix: Mat3, hue_sat: Option<HsvTable>) -> CameraProfile {
        CameraProfile { version: VERSION, model: model.to_owned(), files, samples, matrix: matrix.0, hue_sat, look: None, tone: None }
    }

    /// The name the profile is stored and looked up under: the model, or `<model>@<look>` for a look profile.
    pub fn key(&self) -> String {
        profile_key(&self.model, self.look.as_deref())
    }

    pub fn matrix(&self) -> Mat3 {
        Mat3(self.matrix)
    }

    /// Whether the data is usable (bounded matrix, table shape matching its data).
    fn valid(&self) -> bool {
        let matrix = self.matrix.iter().flatten().all(|v| v.is_finite() && v.abs() < 8.0) && Mat3(self.matrix).inverse().is_some();
        let table = self.hue_sat.as_ref().is_none_or(|t| {
            let dims = (1..=4096).contains(&t.hue_divisions) && (2..=4096).contains(&t.sat_divisions) && (1..=4096).contains(&t.val_divisions);
            let len = t.hue_divisions.checked_mul(t.sat_divisions).and_then(|n| n.checked_mul(t.val_divisions));
            dims && len == Some(t.data.len()) && t.data.iter().flatten().all(|v| v.is_finite()) && t.data.iter().all(|e| e[1] >= 0.0 && e[2] >= 0.0)
        });
        let look = self.look.as_ref().is_none_or(|l| !l.trim().is_empty()) && (self.tone.is_none() || self.look.is_some());
        self.version == VERSION && !self.model.is_empty() && matrix && table && look
    }
}

/// LightCraft's configuration folder (settings, GPU marker, camera profiles); one helper for the app.
pub use crate::config::config_dir;

/// Where camera profiles are read from and written to.
pub fn dir() -> Option<PathBuf> {
    std::env::var_os("LIGHTCRAFT_CAMERA_PROFILES").map(PathBuf::from).or_else(|| config_dir().map(|d| d.join("camera-profiles")))
}

/// File name of `model`'s profile: letters, digits, `-` and `_` kept, anything else `_`.
pub fn file_name(model: &str) -> String {
    let name: String = model.trim().chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    format!("{name}.json")
}

/// `model`, or `<model>@<look>` for a look profile.
fn profile_key(model: &str, look: Option<&str>) -> String {
    match look {
        Some(look) => format!("{}@{}", model.trim(), look.trim()),
        None => model.trim().to_owned(),
    }
}

/// Read and validate a profile file.
pub fn load(path: &Path) -> Result<CameraProfile, String> {
    let size = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if size > MAX_FILE {
        return Err(format!("{}: {size} bytes is too large for a camera profile", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&bytes, &path.display().to_string())
}

/// Parse and validate a profile's JSON (`origin` names it in errors).
fn parse(json: &[u8], origin: &str) -> Result<CameraProfile, String> {
    let profile: CameraProfile = serde_json::from_slice(json).map_err(|e| format!("{origin}: {e}"))?;
    if !profile.valid() {
        return Err(format!("{origin}: not a usable camera profile (version {}, model {:?})", profile.version, profile.model));
    }
    Ok(profile)
}

/// The built-in profile for `model`, if any.
fn bundled(model: &str) -> Option<CameraProfile> {
    let (_, json) = BUNDLED.iter().find(|(m, _)| *m == model)?;
    parse(json.as_bytes(), &format!("built-in profile {model}")).inspect_err(|e| eprintln!("lightcraft: ignoring camera profile {e}")).ok()
}

/// Write `profile` to `dir` (created if needed) as `<model>.json`; returns the path.
pub fn save(profile: &CameraProfile, dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(file_name(&profile.key()));
    let json = serde_json::to_vec_pretty(profile).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

fn cache() -> &'static Mutex<HashMap<String, Option<Arc<CameraProfile>>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<Arc<CameraProfile>>>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// The profile for camera `model`: from [`dir`], else built in; read once per process.
pub fn get(model: &str) -> Option<Arc<CameraProfile>> {
    get_keyed(model, None)
}

/// The look profile `look` (`lightcraft_raw::FilmLook::key`) for camera `model`, from [`dir`]; read once per process.
pub fn get_look(model: &str, look: &str) -> Option<Arc<CameraProfile>> {
    (!look.trim().is_empty()).then(|| get_keyed(model, Some(look))).flatten()
}

fn get_keyed(model: &str, look: Option<&str>) -> Option<Arc<CameraProfile>> {
    let model = model.trim();
    if model.is_empty() {
        return None;
    }
    let key = profile_key(model, look);
    let mut cache = cache().lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(hit) = cache.get(&key) {
        return hit.clone();
    }
    let path = dir().map(|d| d.join(file_name(&key)));
    let profile = match path {
        Some(path) if path.is_file() => match load(&path) {
            Ok(p) if p.key() == key => Some(Arc::new(p)),
            Ok(p) => {
                eprintln!("lightcraft: ignoring camera profile {}: it is for {:?}, not {key:?}", path.display(), p.key());
                None
            }
            Err(e) => {
                eprintln!("lightcraft: ignoring camera profile {e}");
                None
            }
        },
        _ => None,
    };
    let profile = if look.is_none() { profile.or_else(|| bundled(model).map(Arc::new)) } else { profile };
    cache.insert(key, profile.clone());
    profile
}

/// Changes when the profiles folder's contents change (per process): part of the render cache
/// keys, so thumbnails rendered before a profile existed are not reused after.
pub fn cache_key() -> u64 {
    static KEY: OnceLock<u64> = OnceLock::new();
    *KEY.get_or_init(|| {
        let entries = dir().and_then(|d| std::fs::read_dir(d).ok());
        let mut files: Vec<(String, u64, u64)> = entries
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
                Some((e.file_name().to_string_lossy().into_owned(), meta.len(), modified))
            })
            .collect();
        files.sort();
        let mut h = lightcraft_preview::Hasher128::new();
        for (name, len, modified) in &files {
            h.str(name).u64(*len).u64(*modified);
        }
        for (model, json) in BUNDLED {
            h.str(model).str(json);
        }
        h.str(&std::env::var("LIGHTCRAFT_CAMERA_LOOK").unwrap_or_default());
        h.finish().0 as u64
    })
}

/// Per camera model (or model and look, [`Pool::by_look`]): the photos read and their pooled colour pairs.
#[derive(Default)]
pub struct Pool {
    models: HashMap<String, (usize, Vec<([f64; 3], [f64; 3])>)>,
    /// Group photos by model and the in-camera rendering of their JPEG, and fit look profiles (with tone curves).
    by_look: bool,
    /// Per group, each pair's photo exposure scale (`FilmLook::exposure_scale`), for look profiles.
    scales: HashMap<String, Vec<f64>>,
}

/// How close renders of one photo come to its camera JPEG ([`evaluate`]).
pub struct Evaluation {
    /// `<model>@<look>` the camera JPEG was rendered with (empty when the file doesn't say).
    pub shot: String,
    /// Mean CIE76 ΔE of its own look fitted to that JPEG (`None` when the fit is rejected).
    pub own: Option<f64>,
    /// Mean CIE76 ΔE per look profile passed in; NaN for profiles of other models.
    pub looks: Vec<f64>,
}

/// Mean CIE76 ΔE from a photo's camera JPEG ([`crate::camera_preview::evaluate_looks`]) for its own JPEG fit and
/// each of `looks` (look profiles; nothing fitted to this photo). `Ok(None)` when the photo can't be compared.
pub fn evaluate(bytes: &[u8], looks: &[&CameraProfile]) -> Result<Option<Evaluation>, String> {
    let mut raw = lightcraft_raw::decode(bytes).map_err(|e| e.to_string())?;
    raw.opcodes.list3.retain(|op| !op.is_lens_correction());
    let model = raw.metadata.model.as_deref().map(str::trim).unwrap_or_default().to_owned();
    let shot = lightcraft_raw::embedded_preview_film_look(bytes).map(|f| profile_key(&model, Some(&f.key()))).unwrap_or_default();
    let of_model: Vec<&CameraProfile> = looks.iter().copied().filter(|p| p.model.trim() == model).collect();
    let Some((own, des)) = crate::camera_preview::evaluate_looks(&raw, bytes, &of_model) else { return Ok(None) };
    let mut des = des.into_iter();
    let looks = looks.iter().map(|p| if p.model.trim() == model { des.next().unwrap_or(f64::NAN) } else { f64::NAN }).collect();
    Ok(Some(Evaluation { shot, own, looks }))
}

/// Most colour pairs kept per photo, so a few busy photos can't dominate a profile.
const PAIRS_PER_FILE: usize = 4000;

impl Pool {
    /// Add one raw file's colour pairs. `Ok(None)` when the file can't contribute (not an ARW,
    /// NEF or RAF without colour matrices, no usable camera JPEG, too little colour).
    pub fn add(&mut self, bytes: &[u8]) -> Result<Option<String>, String> {
        let mut raw = lightcraft_raw::decode(bytes).map_err(|e| e.to_string())?;
        // colour only: geometric lens corrections would stop the sensor proxy from binning
        raw.opcodes.list3.retain(|op| !op.is_lens_correction());
        let Some(model) = raw.metadata.model.as_deref().map(str::trim).filter(|m| !m.is_empty()) else { return Ok(None) };
        let (key, scale) = if self.by_look {
            let Some(film) = lightcraft_raw::embedded_preview_film_look(bytes) else { return Ok(None) };
            (profile_key(model, Some(&film.key())), film.exposure_scale())
        } else {
            (model.to_owned(), 1.0)
        };
        let Some(pairs) = crate::camera_preview::profile_pairs(&raw, bytes) else { return Ok(None) };
        let step = pairs.len().div_ceil(PAIRS_PER_FILE).max(1);
        let entry = self.models.entry(key.clone()).or_default();
        entry.0 += 1;
        let before = entry.1.len();
        entry.1.extend(pairs.into_iter().step_by(step));
        if self.by_look {
            let added = entry.1.len() - before;
            self.scales.entry(key.clone()).or_default().extend(std::iter::repeat_n(scale, added));
        }
        Ok(Some(key))
    }

    /// A pool that groups photos by model and the in-camera rendering of their JPEG (film simulation and tone
    /// settings, `lightcraft_raw::FilmLook`), and fits look profiles: colour as for a model's profile, plus the
    /// tone and chroma curves of that rendering. Only Fujifilm RAFs record it so far.
    pub fn by_look() -> Pool {
        Pool { by_look: true, ..Pool::default() }
    }

    /// Fit a profile per model with at least `min_files` photos.
    pub fn fit(&self, min_files: usize) -> Vec<Result<CameraProfile, String>> {
        let mut models: Vec<_> = self.models.iter().collect();
        models.sort_by(|a, b| a.0.cmp(b.0));
        models
            .into_iter()
            .filter(|(_, (files, _))| *files >= min_files)
            .map(|(key, (files, pairs))| {
                let (matrix, hue_sat) = crate::camera_preview::fit_profile(pairs).ok_or_else(|| format!("{key}: no usable colour fit"))?;
                let Some((model, look)) = key.split_once('@').filter(|_| self.by_look) else {
                    return Ok(CameraProfile::new(key, *files, pairs.len(), matrix, hue_sat));
                };
                let scales = self.scales.get(key).map(Vec::as_slice).unwrap_or_default();
                let tone = crate::camera_preview::fit_profile_tone(pairs, scales, matrix, hue_sat.clone())
                    .ok_or_else(|| format!("{key}: no usable tone curve"))?;
                let mut profile = CameraProfile::new(model, *files, pairs.len(), matrix, hue_sat);
                profile.look = Some(look.to_owned());
                profile.tone = Some(tone);
                Ok(profile)
            })
            .collect()
    }

    /// Add another pool's photos and pairs (pools filled on separate threads).
    pub fn merge(&mut self, other: Pool) {
        for (model, (files, pairs)) in other.models {
            let entry = self.models.entry(model).or_default();
            entry.0 += files;
            entry.1.extend(pairs);
        }
        for (key, scales) in other.scales {
            self.scales.entry(key).or_default().extend(scales);
        }
    }

    /// Photos read per model.
    pub fn files(&self) -> Vec<(String, usize)> {
        let mut v: Vec<_> = self.models.iter().map(|(m, (n, _))| (m.clone(), *n)).collect();
        v.sort();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> CameraProfile {
        let table = HsvTable { hue_divisions: 4, sat_divisions: 2, val_divisions: 1, data: vec![[5.0, 1.2, 1.0]; 8], srgb_value: false };
        CameraProfile::new("ILCE-7M4", 12, 3456, Mat3([[1.6, -0.5, -0.1], [-0.2, 1.3, -0.1], [-0.1, -0.2, 1.3]]), Some(table))
    }

    #[test]
    fn profiles_round_trip_and_reject_bad_files() {
        let dir = std::env::temp_dir().join(format!("lc-camera-profiles-{}", std::process::id()));
        let path = save(&profile(), &dir).unwrap();
        assert_eq!(path.file_name().unwrap(), "ILCE-7M4.json");
        assert_eq!(load(&path).unwrap(), profile());
        let bad = |p: CameraProfile| {
            let path = dir.join("bad.json");
            std::fs::write(&path, serde_json::to_vec(&p).unwrap()).unwrap();
            load(&path).is_err()
        };
        let mut p = profile();
        p.matrix[0][0] = f64::NAN;
        assert!(bad(p), "non-finite matrix");
        let mut p = profile();
        p.matrix = [[0.0; 3]; 3];
        assert!(bad(p), "singular matrix");
        let mut p = profile();
        if let Some(t) = p.hue_sat.as_mut() {
            t.data.truncate(3);
        }
        assert!(bad(p), "table data doesn't match its shape");
        let mut p = profile();
        p.version = 99;
        assert!(bad(p), "unknown version");
        std::fs::write(dir.join("junk.json"), b"{not json").unwrap();
        assert!(load(&dir.join("junk.json")).is_err());
        assert!(load(&dir.join("missing.json")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bundled_profiles_are_valid_and_named_after_their_model() {
        assert!(!BUNDLED.is_empty());
        for (model, json) in BUNDLED {
            let p = parse(json.as_bytes(), model).unwrap();
            assert_eq!(p.model, *model);
            assert!(p.files >= 5 && p.hue_sat.is_some(), "{model}: {} photos", p.files);
            assert_eq!(bundled(model), Some(p));
        }
        assert!(bundled("No Such Camera").is_none());
    }

    #[test]
    fn file_names_are_safe() {
        assert_eq!(file_name("ILCE-7M4"), "ILCE-7M4.json");
        assert_eq!(file_name(" DSC-RX100M3 "), "DSC-RX100M3.json");
        assert_eq!(file_name("../../etc/passwd"), "______etc_passwd.json");
        assert_eq!(file_name("Ω 1/2"), "__1_2.json");
    }

    #[test]
    fn pool_skips_files_that_cannot_contribute() {
        let mut pool = Pool::default();
        assert!(pool.add(b"not a raw file").is_err());
        assert!(pool.fit(1).is_empty());
        assert!(pool.files().is_empty());
    }
}
