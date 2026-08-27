import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. BELIEF-STATE TRACKER (HIDDEN INFO MITIGATION)
# ==========================================
class LSTMHandPredictor(nn.Module):
    def __init__(self, vocab_size=500, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size) # Probabilitas tipe kartu di tangan

    def predict_hand(self, enemy_discard_log):
        """Memprediksi isi tangan lawan berdasarkan urutan kartu yang dibuang/dimainkan."""
        if not enemy_discard_log:
            return {} # Fallback jika log kosong
            
        # Konversi log ke tensor dan lewati LSTM
        log_tensor = torch.tensor(enemy_discard_log).unsqueeze(0)
        embedded = self.embedding(log_tensor)
        lstm_out, _ = self.lstm(embedded)
        probabilities = torch.softmax(self.fc(lstm_out[:, -1, :]), dim=-1)
        
        # Ekstrak top-k probabilitas kartu ancaman (misal: Boss's Orders, Energy)
        return self.parse_probabilities(probabilities)

# ==========================================
# 2. THREAT ASSESSMENT SENSOR (AGGRO MITIGATION)
# ==========================================
class ThreatAssessmentSensor:
    @staticmethod
    def calculate_lethal_threat(state, belief_state):
        """Menghitung potensi max damage lawan di giliran berikutnya."""
        enemy_active = state.enemy_board.active
        base_damage = enemy_active.get_max_attack_damage()
        
        # Tambahkan potensi damage jika LSTM mendeteksi kartu "Buff" di tangan lawan
        buff_probability = belief_state.get("damage_modifier_card", 0.0)
        expected_buff = 30 if buff_probability > 0.6 else 0
        
        total_threat = base_damage + expected_buff
        return total_threat

# ==========================================
# 3. META-CONTROLLER (MACRO LAYER)
# ==========================================
class MetaControllerPolicy:
    def __init__(self):
        self.q_table = {} # Simpanan policy HRL
        
    def determine_macro_goal(self, lethal_threat, our_active_hp):
        """Sensor Darurat: Override instruksi makro jika nyawa terancam."""
        if lethal_threat >= our_active_hp:
            return "EMERGENCY_EVACUATE"
        return "RESOURCE_DENIAL" # Default strategi Sponge & Hammer

    def calculate_intrinsic_reward(self, previous_state, current_state, belief_state):
        """Mencegah Credit Assignment Problem dengan memberikan Reward Menengah."""
        our_resources = current_state.our_board.total_energy + len(current_state.our_hand)
        enemy_resources = current_state.enemy_board.total_energy + belief_state.get("expected_hand_size", 0)
        
        current_delta = our_resources - enemy_resources
        return float(current_delta * 1.5) # Skala reward

# ==========================================
# 4. TACTICAL CONTROLLER (MICRO LAYER - SPONGE & HAMMER)
# ==========================================
class TacticalController:
    def __init__(self, action_masker):
        self.action_masker = action_masker

    def execute_tactics(self, state, macro_goal, max_steps=15):
        """Mengeksekusi logika mikro berdasarkan instruksi makro dari Meta-Controller."""
        step_count = 0
        action_sequence_log = []
        
        # Action Chunking: Paksa agen menarik kartu di urutan pertama
        action_sequence_log.append(self.force_draw_phase(state))

        while step_count < max_steps:
            legal_moves = self.action_masker.get_valid_moves(state)
            if not legal_moves: break

            if macro_goal == "RESOURCE_DENIAL":
                action = self._execute_sponge_and_hammer(state, legal_moves)
            elif macro_goal == "EMERGENCY_EVACUATE":
                action = self._execute_evacuation(state, legal_moves)
            else:
                action = "PASS"

            if action == "PASS": break
                
            state.apply(action)
            action_sequence_log.append(action)
            step_count += 1
            
        return action_sequence_log

    def _execute_sponge_and_hammer(self, state, legal_moves):
        """Algoritma spesifik untuk menghancurkan energi dan menjebak musuh."""
        # 1. Cari musuh dengan Retreat Cost tertinggi di bench
        enemy_bench = state.enemy_board.bench
        if enemy_bench:
            highest_retreat_enemy = max(enemy_bench, key=lambda x: x.retreat_cost)
            
            # 2. Eksekusi Gusting (Boss's Orders) jika target valid
            if "Boss's Orders" in [m.card_name for m in legal_moves] and highest_retreat_enemy.retreat_cost >= 3:
                return {"type": "PLAY_SUPPORTER", "card": "Boss's Orders", "target": highest_retreat_enemy.id}
        
        # 3. Hancurkan Energi target aktif (Crushing Hammer)
        if "Crushing Hammer" in [m.card_name for m in legal_moves] and state.enemy_board.active.energy > 0:
            return {"type": "PLAY_ITEM", "card": "Crushing Hammer", "target": state.enemy_board.active.id}
            
        # 4. Pasang Energi ke "Sponge" kita jika aman
        if "Attach_Energy" in [m.type for m in legal_moves]:
            return {"type": "ATTACH_ENERGY", "target": state.our_board.active.id}
            
        return "PASS"

    def _execute_evacuation(self, state, legal_moves):
        """Prioritas mundur atau mencari perlindungan."""
        if "Retreat" in [m.type for m in legal_moves]:
            return {"type": "RETREAT", "target": "BENCH_SPONGE"}
        return "PASS"

    def force_draw_phase(self, state):
        state.draw_card()
        return {"type": "DRAW_PHASE", "status": "Forced via Action Chunking"}

# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
class AgentPipeline:
    def __init__(self):
        self.tracker = LSTMHandPredictor()
        self.threat_sensor = ThreatAssessmentSensor()
        self.meta = MetaControllerPolicy()
        self.tactics = TacticalController(action_masker=None) # Disuntikkan via engine

    def run_turn(self, raw_state, enemy_discard_log):
        # 1. Update Belief State
        belief_state = self.tracker.predict_hand(enemy_discard_log)
        
        # 2. Threat Assessment
        lethal_threat = self.threat_sensor.calculate_lethal_threat(raw_state, belief_state)
        our_hp = raw_state.our_board.active.current_hp
        
        # 3. Meta-Controller menentukan tujuan
        macro_goal = self.meta.determine_macro_goal(lethal_threat, our_hp)
        
        # 4. Tactical Controller mengeksekusi dengan batasan 15 step
        turn_log = self.tactics.execute_tactics(raw_state, macro_goal, max_steps=15)
        
        # 5. Hitung Reward Intrinsic untuk pembelajaran berkelanjutan
        reward = self.meta.calculate_intrinsic_reward(raw_state, raw_state, belief_state)
        
        return turn_log, reward