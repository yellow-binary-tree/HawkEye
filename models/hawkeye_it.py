import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

from .blip2.blip2 import Blip2Base, disabled_train
from transformers import LlamaTokenizer, LlamaConfig

logger = logging.getLogger(__name__)


class HawkEye_it(Blip2Base):
    """
    HawkEye model.
    """
    def __init__(self, config):
        super().__init__()
        # pretrained_path
        vit_blip_model_path = config.get("vit_blip_model_path", None)
        llama_model_path = config.get("llama_model_path")
        stage3_model_path = config.get("stage3_model_path", "")  
        self.bert_path = config.get("bert_path", "bert-base-uncased")
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # qformer
        num_query_token = config.get("num_query_token")
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        self.qformer_text_input = config.get("qformer_text_input", True)
        # prompt
        max_txt_len = config.get("max_txt_len", 32)
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.start_token = config.get("start_token", "<Video>")
        self.end_token = config.get("end_token", "</Video>")
        self.img_start_token = config.get("img_start_token", "<Image>")
        self.img_end_token = config.get("img_end_token", "</Image>")
        logger.info(f"Add instruction in qformer: {self.qformer_text_input}")
        self.num_frame_tokens = config.get("num_frame_tokens", 0)
        # debug
        debug = config.get("debug", False)
        use_flash_attention = config.get("use_flash_attention", False)
        self.use_lora = config.get("use_lora", False)
        lora_r = config.get("lora_r", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.05)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.low_resource = low_resource
        self.vision_encoder, self.vision_layernorm, = self.init_vision_encoder_umt(config)
        self.qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.num_frame_tokens, config.vision_encoder.encoder_embed_dim,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )
        
        if not self.qformer_text_input:
            self.qformer.bert.embeddings.word_embeddings = None
            self.qformer.bert.embeddings.position_embeddings = None
            for layer in self.qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.qformer.resize_token_embeddings(len(self.tokenizer))
        self.qformer.cls = None

        if vit_blip_model_path:
            logger.info(f"Load ViT and QFormer from {vit_blip_model_path}")
            state_dict = torch.load(vit_blip_model_path, map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info('Loading ViT and Q-Former Done')
        
        self.extra_num_query_token = extra_num_query_token
        if extra_num_query_token > 0:
            logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
            self.extra_query_tokens = nn.Parameter(
                torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
            )

        if freeze_vit:
            logger.info("freeze vision encoder")
            for _, param in self.vision_encoder.named_parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
            self.vision_encoder.train = disabled_train
            for _, param in self.vision_layernorm.named_parameters():
                param.requires_grad = False
            self.vision_layernorm = self.vision_layernorm.eval()
            self.vision_layernorm.train = disabled_train

        if freeze_qformer:
            logger.info("freeze Qformer")
            for _, param in self.qformer.named_parameters():
                param.requires_grad = False
            self.qformer = self.qformer.eval()
            self.qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        logger.info('Loading LLAMA')
        # problem: do we need to set truncation_side="left"?
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if use_flash_attention:
            # logger.info("Use flash attention")
            # from .blip2.modeling_llama_mem import LlamaForCausalLM
            raise NotImplementedError("LlamaForCausalLM with flash attention is deprecated when adding frame tokens")
        else:
            from .blip2.modeling_llama import LlamaForCausalLM
        llama_config = LlamaConfig.from_pretrained(llama_model_path)
        llama_config.num_frame_tokens = self.num_frame_tokens

        if debug:
            logger.info("Debug mode, build small LLAMA")
            llama_config.hidden_size = 4096
            llama_config.intermediate_size = 2048
            llama_config.num_attention_heads = 8
            llama_config.num_hidden_layers = 4
            llama_config.torch_dtype = torch.float16
            self.llama_model = LlamaForCausalLM(llama_config)
        else:
            logger.info("loading LLAMA ckpt from %s" % llama_model_path)
            if self.low_resource:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path, config=llama_config,
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map="auto",
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model_path, config=llama_config,
                    torch_dtype=torch.float16,
                )

        llama_config.vocab_size = 32000
        self.llama_model.resize_token_embeddings(llama_config.vocab_size)       # this will change the vocab_size attribute in llama_model
        logger.info(f'llama config: {llama_config}')

        if self.num_frame_tokens:
            logger.info('adding %d special frame tokens to model.llama_tokenizer and self.tokenizer' % self.num_frame_tokens)
            self.llama_tokenizer.add_tokens(["<loc>"] + ["<frame%d>" % i for i in range(self.num_frame_tokens)] + ["</loc>"])
            self.tokenizer.add_tokens(["<loc>"] + ["<frame%d>" % i for i in range(self.num_frame_tokens)] + ["</loc>"])

        logger.info("freeze LLAMA")
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading LLAMA Done')

        if self.use_lora:
            logger.info("Use lora")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        else:
            logger.info("freeze LLAMA")
            for _, param in self.llama_model.named_parameters():
                param.requires_grad = False

        self.llama_proj = nn.Linear(
            self.qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len

        if self.num_frame_tokens:
            for name, param in self.llama_model.named_parameters():
                if "frame_lm_head" in name or "frame_embed_tokens" in name:
                    logger.info("do not freeze %s" % name)
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True

        # load it model weights
        if stage3_model_path:
            logger.info(f"Load stage 3 HawkEye from: {stage3_model_path}")
            ckpt = torch.load(stage3_model_path, map_location="cpu")
            if 'model' in ckpt.keys():
                msg = self.load_state_dict(ckpt['model'], strict=False)
            else:
                msg = self.load_state_dict(ckpt, strict=False)
            logger.info(msg)
        self.devices = [torch.device(0), torch.device(0)]       # device name of other params / vision encoder


    def set_device_ids(self, device_ids):
        # used to put model on different devices when GPU memory is small, optional
        assert isinstance(device_ids, list) and 1 <= len(device_ids) <= 2
        if len(device_ids) == 1:
            device_ids = [device_ids[0], device_ids[0]]
        logger.info('using device {} for model'.format(device_ids))
        self.devices = [torch.device(i) for i in device_ids]

        self.vision_encoder.to(self.devices[1])         # to many parameters to tune in vit, maybe put it on another gpu
        self.vision_layernorm.to(self.devices[1])
        self.qformer.to(self.devices[0])
        self.query_tokens = nn.Parameter(self.query_tokens.to(self.devices[0]))
        self.extra_query_tokens = nn.Parameter(self.extra_query_tokens.to(self.devices[0]))
        self.llama_proj.to(self.devices[0])
        self.llama_model.to(self.devices[0])

    def vit_to_cpu(self):
        self.vision_layernorm.to("cpu")
        self.vision_layernorm.float()
        self.vision_encoder.to("cpu")
        self.vision_encoder.float()

    def encode_img(self, image, instruction):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            T = image.shape[1]
            use_image = True if T == 1 else False
            image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

            image = image.to(self.devices[1])
            image_embeds = self.vision_encoder(image, use_image)        # vision encoder may on another device
            B, T, L, C = image_embeds.shape
            image_embeds = image_embeds.reshape(B, -1, C)
            image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            if self.extra_num_query_token > 0:
                query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            else:
                query_tokens = self.query_tokens
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.qformer_text_input:
                text_Qformer = self.tokenizer(
                    instruction,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image_embeds.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

                query_output = self.qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llama = self.llama_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        return inputs_llama, use_image
        
    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def forward(self, image, text_input, instruction):
        img_embeds, use_image = self.encode_img(image, instruction)
        batch_size, img_len, _ = img_embeds.shape

        # mark the largest length
        # when padding, the attention mask will be 0
        max_len = 0
        input_embed_list = []
        p_before_len_list = []
        target_list = []
        # handle each prompt individually

        for idx, prompt in enumerate(text_input):
            tmp_img_embeds = img_embeds[idx].unsqueeze(0)
            # split the prompt via END_TOKEN
            end_token = self.img_end_token if use_image else self.end_token
            p_before, p_after = prompt.split(end_token)
            p_after = end_token + p_after
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(tmp_img_embeds.device)

            if self.use_lora:
                p_before_embeds = self.llama_model.base_model.model.model.embed_tokens_(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.base_model.model.model.embed_tokens_(p_after_tokens.input_ids)
            else:
                p_before_embeds = self.llama_model.model.embed_tokens_(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens_(p_after_tokens.input_ids)
            input_embeds = torch.cat([p_before_embeds, tmp_img_embeds, p_after_embeds], dim=1)

            # extract the answers and mask the target
            # the answers are only in the p_after
            sep1 = self.begin_signal + self.role[0] + ": "
            sep2 = self.begin_signal + self.role[1] + ": "

            if sep2 in p_after:
                raw_text = p_after.split(sep2)
                for idx in range(1, len(raw_text)):
                    raw_text[idx] = sep2 + raw_text[idx]
                # the first raw_text contains system and question
                # the last raw_text only contains answer
                # rstrip() for the extra " "
                answer_targets = p_after_tokens.input_ids.clone()
                # target: "###Human:       ###Assistant: xxxxx. ###"
                system = raw_text[0].split(sep1)[0]
                system_len = self._get_text_len(system.rstrip())
                sep_len = self._get_text_len(sep1.rstrip())
                cur_len = self._get_text_len(raw_text[0].rstrip())
                answer_targets[:, :system_len] = -100
                answer_targets[:, (system_len+sep_len):cur_len] = -100
                for text in raw_text[1:-1]:
                    total_len = self._get_text_len(text.rstrip())
                    ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                    answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                    cur_len += total_len
                cur_len += self._get_text_len(raw_text[-1].rstrip())
                assert cur_len == answer_targets.shape[1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {prompt}"

                max_len = max(max_len, input_embeds.shape[1])
                input_embed_list.append(input_embeds)
                p_before_len_list.append(p_before_tokens.input_ids.shape[1])
                target_list.append(answer_targets)

            else:
                # no "###Assistant: " in string. Which indicates the input is not a chat conversation, but just plain text for auto-regressive generation
                max_len = max(max_len, input_embeds.shape[1])
                input_embed_list.append(input_embeds)
                p_before_len_list.append(p_before_tokens.input_ids.shape[1])
                answer_targets = p_after_tokens.input_ids.clone()
                answer_targets[:, : self._get_text_len(end_token)] = -100   # </Video> or </Image>, we do noet need to predict this
                target_list.append(answer_targets)

        # plus one for bos
        # max_txt_len plus num_query_token is the max len
        txt_len = min(max_len + 1, self.max_txt_len + img_len)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device) * self.llama_tokenizer.pad_token_id
        if self.use_lora:
            inputs_embeds = self.llama_model.base_model.model.model.embed_tokens_(inputs_embeds)
        else:
            inputs_embeds = self.llama_model.model.embed_tokens_(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(img_embeds.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(img_embeds.device).fill_(-100)
        # set bos_token
        inputs_embeds[:, :1] = self.llama_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            # if less than txt_len, the input will be padding
            # if more than txt_len, the input will be truncated
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            # the attention_mask is 0 when padding
            attention_mask[idx, :(input_len+1)] = 1
            # the target is -100 when padding
            p_before_len = p_before_len_list[idx]
            targets[idx, (p_before_len+img_len+1):(input_len+1)] = target_list[idx][0, :(input_len-p_before_len-img_len)]

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
    
        return dict(
            loss=outputs.loss,
        )
