export type ClassProbability = {
  class_name: string;
  probability: number;
};

export type UploadScanResult = {
  patient_id: string;
  patient_name: string;
  scan_id: number;
  scan_date: string;
  tumor_detected: boolean;
  tumor_type: string | null;
  confidence_score: number;
  tumor_area: number;
  tumor_volume: number | null;
  risk_category: string;
  uncertainty_warning: string | null;
  progression_status: string;
  explainability_consistency_score: number | null;
  explainability_warning: string | null;
  longitudinal_tumor_progression_index: number | null;
  model_version: string;
  runtime_mode: string;
  runtime_note: string | null;
  class_probabilities: ClassProbability[];
  report_path: string;
  gradcam_path: string | null;
  mask_path: string | null;
  overlay_path: string | null;
  image_url: string | null;
  report_url: string | null;
  gradcam_url: string | null;
  mask_url: string | null;
  overlay_url: string | null;
  disclaimer: string;
  stage_note: string;
  attribution: string;
};

export type ScanCore = {
  id: number;
  patient_db_id: number;
  scan_date: string;
  image_path: string;
  mask_path: string | null;
  gradcam_path: string | null;
  report_path: string | null;
  overlay_path: string | null;
  tumor_detected: boolean;
  tumor_type: string | null;
  confidence_score: number;
  tumor_area: number;
  tumor_volume: number | null;
  risk_category: string;
  explainability_consistency_score: number | null;
  model_version: string;
  radiologist_notes: string | null;
  created_at: string;
};

export type ScanAssetBundle = {
  image_url: string | null;
  mask_url: string | null;
  gradcam_url: string | null;
  overlay_url: string | null;
  report_url: string | null;
};

export type PatientScanPayload = {
  scan: ScanCore;
  class_probabilities: ClassProbability[];
  assets: ScanAssetBundle;
};

export type PatientProfilePayload = {
  patient: {
    id: number;
    patient_id: string;
    patient_code: string | null;
    name: string;
    age: number;
    gender: string;
    contact: string | null;
    created_at: string;
  };
  scans: PatientScanPayload[];
};

export type CompareScansResponse = {
  patient_id: string;
  previous_scan_id: number;
  current_scan_id: number;
  previous_scan_date: string;
  current_scan_date: string;
  previous_tumor_area: number;
  current_tumor_area: number;
  previous_tumor_volume: number | null;
  current_tumor_volume: number | null;
  absolute_change: number;
  percentage_change: number;
  tumor_type_change: string;
  confidence_difference: number;
  previous_risk_level: string | null;
  current_risk_level: string | null;
  risk_level_change: string | null;
  progression_status: string;
  longitudinal_tumor_progression_index: number | null;
  summary: string;
  previous_scan_assets: ScanAssetBundle | null;
  current_scan_assets: ScanAssetBundle | null;
  progression_chart_url: string | null;
};

export type ModelMetric = {
  id: number;
  model_name: string;
  task_type: string;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  auc: number | null;
  dice: number | null;
  iou: number | null;
  hausdorff95: number | null;
  inference_time: number | null;
  training_time: number | null;
  model_size: number | null;
  best_use_case: string | null;
  status?: string;
};
