import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

def analyze_decode_attention_layers(outputs, tokenizer=None, step=0, save_path=None, max_display_tokens=50):
    """
    Analyze the attention heatmap for each layer during the decode stage.
    
    Args:
        outputs: Model output, which should contain attentions.
        tokenizer: Tokenizer used to decode tokens (optional).
        step: Decode step index.
        save_path: Save path (optional).
        max_display_tokens: Maximum number of tokens to display.
    
    Returns:
        dict: Dictionary containing attention analysis results for each layer.
    """
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        print(f"Decode Step {step}: No attention weights found")
        return None
    
    # Get attention weights for all layers.
    all_layers_attention = outputs.attentions
    num_layers = len(all_layers_attention)
    
    print(f"\n=== Decode Step {step} - All Layers Attention Analysis ===")
    print(f"Total layers: {num_layers}")
    
    # Prepare the result dictionary.
    layers_data = {}
    
    # Calculate the grid layout.
    cols = min(4, num_layers)  # Up to 4 subplots per row.
    rows = (num_layers + cols - 1) // cols  # Round up.
    
    # Create a large figure showing attention heatmaps for all layers.
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for layer_idx, layer_attention in enumerate(all_layers_attention):
        # layer_attention shape: [batch, heads, seq_len, seq_len]
        
        # Average across all attention heads.
        attention_mean = layer_attention.mean(dim=1)[0]  # shape: [seq_len, seq_len]
        seq_len = attention_mean.shape[0]
        
        # Analyze attention from the last position (newly generated token) to all positions.
        last_pos_attention = attention_mean[-1, :]  # Attention from the last token to all tokens.
        
        # Find the top attention scores.
        top_k = min(10, len(last_pos_attention))
        top_values, top_indices = torch.topk(last_pos_attention, top_k)
        
        print(f"\nLayer {layer_idx + 1}:")
        print(f"  Attention shape: {attention_mean.shape}")
        print(f"  Top {top_k} attention positions:")
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            token_text = ""
            if tokenizer and idx.item() < seq_len:
                # Note: token retrieval here may need to depend on the specific input.
                try:
                    # Here we assume token information can be obtained somehow.
                    token_text = f" (pos: {idx.item()})"
                except:
                    token_text = f" (pos: {idx.item()})"
            print(f"    {i+1}. Position {idx.item()}: {val.item():.6f}{token_text}")
        
        # Store layer data.
        layers_data[f'layer_{layer_idx}'] = {
            'attention_matrix': attention_mean,
            'last_pos_attention': last_pos_attention,
            'top_positions': top_indices.cpu().numpy(),
            'top_values': top_values.cpu().float().numpy(),
            'layer_index': layer_idx
        }
        
        # Draw the attention heatmap for the current layer.
        ax = axes[layer_idx] if layer_idx < len(axes) else None
        if ax is not None:
            # Limit the number of displayed tokens for readability.
            display_size = min(max_display_tokens, attention_mean.shape[0])
            
            # Convert to float32 to avoid bfloat16 issues.
            attention_matrix = attention_mean.cpu().float().numpy()[:display_size, :display_size]
            
            # Draw the heatmap.
            sns.heatmap(attention_matrix, cmap='Blues', cbar=True, ax=ax,
                       square=True, cbar_kws={'shrink': 0.8})
            ax.set_title(f'Layer {layer_idx + 1} - Attention Matrix')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Set axis ticks.
            if display_size <= 20:
                ax.set_xticks(range(0, display_size, 2))
                ax.set_yticks(range(0, display_size, 2))
            else:
                ax.set_xticks(range(0, display_size, 10))
                ax.set_yticks(range(0, display_size, 10))
    
    # Hide unused subplots.
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Decode Step {step} - All Layers Attention Heatmaps', fontsize=16)
    plt.tight_layout()
    
    # Save the figure.
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/decode_step_{step}_all_layers_attention.png", 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Separately analyze detailed information for the last layer.
    if num_layers > 0:
        last_layer_data = layers_data[f'layer_{num_layers-1}']
        
        # Create a detailed analysis figure for the last layer.
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Subplot 1: full attention matrix of the last layer.
        display_size = min(max_display_tokens, last_layer_data['attention_matrix'].shape[0])
        attention_matrix = last_layer_data['attention_matrix'].cpu().float().numpy()[:display_size, :display_size]
        sns.heatmap(attention_matrix, cmap='Blues', cbar=True, ax=axes[0])
        axes[0].set_title(f'Layer {num_layers} - Full Attention Matrix')
        axes[0].set_xlabel('Key Position')
        axes[0].set_ylabel('Query Position')
        
        # Subplot 2: attention distribution of the last position.
        last_pos_attention = last_layer_data['last_pos_attention']
        plot_len = min(100, len(last_pos_attention))
        attention_plot = last_pos_attention[:plot_len].cpu().float().numpy()
        axes[1].plot(attention_plot)
        axes[1].set_title(f'Layer {num_layers} - Last Token Attention Distribution')
        axes[1].set_xlabel('Position')
        axes[1].set_ylabel('Attention Score')
        axes[1].grid(True)
        
        # Subplot 3: top attention positions.
        top_values_plot = last_layer_data['top_values']
        axes[2].bar(range(len(top_values_plot)), top_values_plot)
        axes[2].set_title(f'Layer {num_layers} - Top Attention Scores')
        axes[2].set_xlabel('Rank')
        axes[2].set_ylabel('Attention Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/decode_step_{step}_last_layer_detailed.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    return layers_data

def calculate_question_frame_attention_scores(model_output, image_bounds, question_bound):
    """
    Calculate attention scores from question tokens to each video frame.
    
    Args:
        model_output: Model forward output containing attentions.
        image_bounds: Frame-token mapping list from get_image_bound.
        question_bound: Start/end token position tuple for the question (start_idx, end_idx).
    
    Returns:
        dict: Attention scores for each frame in each layer.
        Structure: {layer_idx: {frame_id: attention_score}}
    """
    if not hasattr(model_output, 'attentions') or model_output.attentions is None:
        raise ValueError("模型输出中没有attention信息，请确保在forward时设置output_attentions=True")
    
    attentions = model_output.attentions
    
    all_layers_scores = []
    
    # Iterate over each layer.
    for layer_idx, layer_attention in enumerate(attentions):
        # Get attention weights for the specified layer [batch_size, num_heads, seq_len, seq_len].
        attention_weights = layer_attention
        if attention_weights.dtype == torch.bfloat16:
            attention_weights = attention_weights.float()
        # Assume batch_size=1 and take the first sample.
        attention_weights = attention_weights[0]  # [num_heads, seq_len, seq_len]
        # Average across all attention heads.
        avg_attention = attention_weights.mean(dim=0)  # [seq_len, seq_len]
        
        frame_attention_scores = []
        question_start, question_end = question_bound
        
        # Iterate over each frame and calculate its attention score.
        for frame_id, image_bound in enumerate(image_bounds):
            frame_start = image_bound[0]
            frame_end = image_bound[1]
            # Calculate the mean attention score from question tokens to this frame's tokens.
            question_to_frame_attention = avg_attention[question_start:question_end+1, frame_start:frame_end+1].mean().item()
            frame_attention_scores.append(question_to_frame_attention)
        
        all_layers_scores.append(frame_attention_scores)
    
    return all_layers_scores

def draw_question_frame_attention_heatmap(output_path, all_layers_scores, scene_index_list):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    layer_indices = sorted(all_layers_scores.keys())
    num_layers = len(layer_indices)
    
    # Build the score matrix: shape [num_frames, num_layers].
    frame_ids = []
    if all_layers_scores:
        frame_ids = sorted(all_layers_scores[layer_indices[0]].keys())
    score_matrix = []
    for frame_id in frame_ids:
        row = []
        for layer_id in layer_indices:
            score = all_layers_scores[layer_id][frame_id]
            row.append(score)
        score_matrix.append(row)
    score_matrix = np.array(score_matrix)

    plt.figure(figsize=(1.5*num_layers+2, 0.5*len(frame_ids)+4))
    im = plt.imshow(score_matrix, aspect='auto', cmap='YlGnBu')

    plt.colorbar(im, shrink=0.8, label='Attention Score')
    plt.xlabel('Layer')
    plt.ylabel('Frame Index')
    plt.xticks(range(len(layer_indices)), [f'L{lid}' for lid in layer_indices])
    plt.yticks(range(len(frame_ids)), [f'F{fid}' for fid in frame_ids])
    
    # Mark scene boundaries.
    scene_boundaries = []
    current_scene = scene_index_list[0]
    for idx, scene_id in enumerate(scene_index_list):
        if scene_id != current_scene:
            scene_boundaries.append(idx - 0.5)
            current_scene = scene_id
    for boundary in scene_boundaries:
        plt.axhline(y=boundary, color='red', linestyle='--', linewidth=1)

    plt.title('Question to Frame Attention Scores Heatmap')
    plt.tight_layout()
    
    # Save the heatmap to a file.
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Question to Frame Attention热力图已保存到: {output_path}")
    plt.close()

def calculate_frame_attention_scores(model_output, image_bounds, layer_indices=None):
    """
    Calculate attention scores for each video frame, supporting all layers or selected layers.
    
    Args:
        model_output: Model forward output containing attentions.
        frame_token_map: Frame-token mapping dictionary from analyze_video_frame_tokens.
        layer_indices: List of layer indices to analyze; None means all layers, and -1 means only the last layer.
    
    Returns:
        dict: Attention scores for each frame in each layer.
        Structure: {layer_idx: {frame_id: {attention_scores}}}
    """
    if not hasattr(model_output, 'attentions') or model_output.attentions is None:
        raise ValueError("模型输出中没有attention信息，请确保在forward时设置output_attentions=True")
    # scene_index_list.insert(0, 0)
    attentions = model_output.attentions
    
    # Determine which layers to analyze.
    if layer_indices is None:
        layer_indices = list(range(len(attentions)))  # Analyze all layers.
    elif layer_indices == -1:
        layer_indices = [len(attentions) - 1]  # Analyze only the last layer.
    elif isinstance(layer_indices, int):
        layer_indices = [layer_indices]  # Convert a single layer into a list.
    
    all_layers_scores = {}
    scene_to_scene_attention_all_layers = {}
    import numpy as np
    # Iterate over each layer.
    for layer_idx in layer_indices:
        # Get attention weights for the specified layer [batch_size, num_heads, seq_len, seq_len].
        attention_weights = attentions[layer_idx]
        if attention_weights.dtype == torch.bfloat16:
            attention_weights = attention_weights.float()
        # Assume batch_size=1 and take the first sample.
        attention_weights = attention_weights[0]  # [num_heads, seq_len, seq_len]
        # Average across all attention heads.
        avg_attention = attention_weights.mean(dim=0)  # [seq_len, seq_len]
        frame_attention_scores = {}
        # Iterate over each frame and calculate its attention score.
        for frame_id, image_bound in enumerate(image_bounds):
            frame_start = image_bound[0]
            frame_end = image_bound[1]
            # Calculate the mean attention score for this frame's tokens.
            last_token_idx = avg_attention.shape[0] - 1
            frame_attention = avg_attention[last_token_idx, frame_start:frame_end+1].mean().item()
            frame_internal_attention = avg_attention[frame_start:frame_end+1, frame_start:frame_end+1].mean().item()
            other_to_frame_attention = avg_attention[:, frame_start:frame_end+1].mean().item()
            # Sum the attention scores from tokens in other frames to all tokens in the current frame.
            other_frames_to_this_frame_attention = []
            for other_frame_id, other_image_bound in enumerate(image_bounds):
                other_start = other_image_bound[0]
                other_end = other_image_bound[1]
                attn_sum = avg_attention[frame_start:frame_end+1, other_start:other_end+1].sum().item()
                other_frames_to_this_frame_attention.append({
                    'other_frame_id': other_frame_id,
                    'attn_sum': attn_sum,
                    'other_token_range': (other_start, other_end)
                })
            frame_attention_scores[frame_id] = {
                'last_token_to_frame': frame_attention,
                'frame_internal_attention': frame_internal_attention,
                'others_to_frame_attention': other_to_frame_attention,
                'other_frames_to_this_frame_attention': other_frames_to_this_frame_attention,
                'token_range': (frame_start, frame_end)
            }
        all_layers_scores[layer_idx] = frame_attention_scores

    return all_layers_scores

def draw_frame_attention_heatmap(output_path, all_layers_scores, scene_index_list):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    layer_indices = sorted(all_layers_scores.keys())
    num_layers = len(layer_indices)
    
    # Build the score matrix: shape [num_frames, num_layers].
    frame_ids = []
    if all_layers_scores:
        frame_ids = sorted([fid for fid in all_layers_scores[layer_indices[0]].keys() if fid != 'question'])
    for layer_id in layer_indices:
        score_matrix = []
        
        for frame_id in frame_ids:
            row = [item['attn_sum'] for item in all_layers_scores[layer_id][frame_id]['other_frames_to_this_frame_attention']]
            score_matrix.append(row)

        score_matrix = np.array(score_matrix)

        plt.figure(figsize=(1.5*num_layers+2, 0.5*len(frame_ids)+4))
        im = plt.imshow(score_matrix, aspect='auto', cmap='YlGnBu')

        plt.colorbar(im, shrink=0.8, label='Attention Score')
        plt.xlabel('Frame Index')
        plt.ylabel('Frame Index')
        plt.yticks(range(len(frame_ids)), [f'F{fid}' for fid in frame_ids])
        plt.xticks(range(len(frame_ids)), [f'F{fid}' for fid in frame_ids])

        # Mark scene boundaries.
        scene_boundaries = []
        current_scene = scene_index_list[0]
        for idx, scene_id in enumerate(scene_index_list):
            if scene_id != current_scene:
                scene_boundaries.append(idx - 0.5)
                current_scene = scene_id
        for boundary in scene_boundaries:
            plt.axhline(y=boundary, color='red', linestyle='--', linewidth=1)

        plt.title('Frame Attention Scores Heatmap with Scene Boundaries')
        plt.tight_layout()
        
        # Save the heatmap to a file.
        heatmap_file = os.path.join(output_path, f"frame_attention_scores_heatmap_with_scenes_{layer_id}.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"Frame Attention热力图已保存到: {heatmap_file}")
        plt.close()

def draw_attention_scores(output_path, all_layers_scores, important_frame):
    # Output attention scores to a CSV file.
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = os.path.join(output_path, "frame_attention_scores.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    import csv
    import numpy as np
    
    # Get IDs for all layers and frames.
    layer_ids = sorted(all_layers_scores.keys())
    frame_ids = []
    if all_layers_scores:
        frame_ids = sorted([fid for fid in all_layers_scores[layer_ids[0]].keys() if fid != 'question'])
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write the header: Frame_ID, Layer_0, Layer_1, ..., Layer_N.
        header = ['Frame_ID'] + [f'Layer_{layer_id}' for layer_id in layer_ids]
        writer.writerow(header)
        
        # Write data for each frame.
        for frame_id in frame_ids:
            row = [f'Frame_{frame_id}']
            for layer_id in layer_ids:
                if frame_id in all_layers_scores[layer_id]:
                    score = all_layers_scores[layer_id][frame_id]['others_to_frame_attention']
                    row.append(f"{score:.6f}")
                else:
                    row.append("N/A")
            writer.writerow(row)
    print(f"Attention分数已保存到CSV文件: {output_file}")
    
    important_output_file = os.path.join(output_path, "important_frame_attention_scores.csv")
    with open(important_output_file, 'w', newline='', encoding='utf-8') as f_imp:
        writer_imp = csv.writer(f_imp)
        header_imp = ['Frame_ID'] + [f'Layer_{layer_id}' for layer_id in layer_ids]
        writer_imp.writerow(header_imp)
        for frame_id in frame_ids:
            if int(frame_id) in important_frame:
                row = [f'Frame_{frame_id}']
                for layer_id in layer_ids:
                    if frame_id in all_layers_scores[layer_id]:
                        score = all_layers_scores[layer_id][frame_id]['others_to_frame_attention']
                        row.append(f"{score:.6f}")
                    else:
                        row.append("N/A")
                writer_imp.writerow(row)
    print(f"重要帧Attention分数已保存到CSV文件: {important_output_file}")

    import matplotlib.pyplot as plt

    # Build the score matrix: shape [num_frames, num_layers].
    score_matrix = []
    for frame_id in frame_ids:
        row = []
        for layer_id in layer_ids:
            if frame_id in all_layers_scores[layer_id]:
                score = all_layers_scores[layer_id][frame_id]['others_to_frame_attention']
                row.append(score)
            else:
                row.append(np.nan)
        score_matrix.append(row)
    score_matrix = np.array(score_matrix)

    plt.figure(figsize=(1.5*len(layer_ids)+2, 0.5*len(frame_ids)+4))
    im = plt.imshow(score_matrix, aspect='auto', cmap='YlGnBu')

    plt.colorbar(im, shrink=0.8, label='Attention Score')
    plt.xlabel('Layer')
    plt.ylabel('Frame Index')
    plt.xticks(range(len(layer_ids)), [f'L{lid}' for lid in layer_ids])
    plt.yticks(range(len(frame_ids)), [f'F{fid}' for fid in frame_ids])

    # Mark important frames.
    for idx, frame_id in enumerate(frame_ids):
        if int(frame_id) in important_frame:
            plt.gca().add_patch(
                plt.Rectangle(
                    (-0.5, idx-0.5), len(layer_ids), 1, fill=False, edgecolor='red', linewidth=2
                )
            )
            plt.text(len(layer_ids)-0.5, idx, '★', color='red', va='center', ha='right', fontsize=14)

    plt.title('Frame Attention Scores Heatmap')
    plt.tight_layout()
    # Save the heatmap to a file.
    heatmap_file = os.path.join(output_path, "frame_attention_scores_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"Attention热力图已保存到: {heatmap_file}")
    
