import torch
import torch.nn as nn
import os
import onnx

WINDOW = 250  # Context Window Size

class BareTransformer(nn.Module):
    def __init__(self, input_dim=36, d_model=12, nhead=3, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dim_feedforward=32
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, WINDOW, d_model))
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [Batch, 250, 36]
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)

def convert_pt_to_onnx(pt_filepath="c:/Users/user/OneDrive/Desktop/KOSPIWED/best_model_done.pt", output_onnx="c:/Users/user/OneDrive/Desktop/KOSPIWED/model.onnx"):
    if not os.path.exists(pt_filepath):
        print(f"❌ 에러: '{pt_filepath}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
        return

    print(f"🔄 '{pt_filepath}' 로딩 중...")
    
    # 1. 모델 객체 생성 (36개 피처 기준)
    model = BareTransformer(input_dim=36)
    
    # 2. PyTorch 가중치(state_dict) 로드
    state_dict = torch.load(pt_filepath, map_location='cpu')
    
    # state_dict 구조 분기 처리
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    model.load_state_dict(state_dict)
    model.eval()  # 추론 모드 전환
    print("✅ 모델 가중치 로드 완료!")

    # 3. ONNX 트레이싱용 임시 입력 [Batch: 1, Sequence: 250, Features: 36]
    tracing_input = torch.zeros(1, WINDOW, 36)

    # 기존 생성된 .onnx 및 .data 잔여 파일 삭제
    if os.path.exists(output_onnx):
        os.remove(output_onnx)
    if os.path.exists(output_onnx + ".data"):
        os.remove(output_onnx + ".data")

    # 4. ONNX 내보내기 (단일 파일 강제 분리 옵션 해제)
    print(f"📦 '{output_onnx}' 생성 중...")
    
    # 임시 파일 경로
    temp_onnx = "temp_model.onnx"
    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)
        
    torch.onnx.export(
        model,
        tracing_input,
        temp_onnx,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],       # 프론트엔드 호환 입력명
        output_names=['output']      # 프론트엔드 호환 출력명
    )
    
    # 5. 외부 .data 파일 없이 완전 단일(Single File) 바이너리로 가중치 팩킹 후 저장
    onnx_model = onnx.load(temp_onnx)
    onnx.save_model(
        onnx_model,
        output_onnx,
        save_as_external_data=False  # 외부 파일 생성 금지 (단일 온디바이스 바이너리 통합)
    )

    # 임시 변환 파일 정리
    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)
    if os.path.exists(temp_onnx + ".data"):
        os.remove(temp_onnx + ".data")
        
    print("==========================================")
    print(f"🎉 성공! 가중치가 완전히 내장된 통합 파일 생성 완료: {os.path.abspath(output_onnx)}")
    print("==========================================")

if __name__ == "__main__":
    convert_pt_to_onnx()