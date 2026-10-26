// Licensed under the MIT License
// https://github.com/craigahobbs/ollama-chat/blob/main/LICENSE

// The Ollama Chat voice playback BareScript runtime functions - text read aloud with PicoVox speech synthesis and Web
// Audio. The playback's text may grow as it plays (a model response that's still generating): the text's last sentence
// is held back until the text is final, and if playback reaches the end of the text before it's final, it pauses.

import {defaultSampleRate, modelLoader, paragraphPause, sentenceBlocks, sentencePause} from './picovox/lib/synthesis.js';
import {speechText, splitSentences} from './picovox/lib/sentences.js';
import {ValueArgsError, valueArgsModel, valueArgsValidate} from './markdown-up/bare-script/lib/value.js';
import {voices} from './picovox/lib/index.js';


// The PicoVox models - loaded on first use, once
const loadModel = modelLoader(() => import('./picovox/lib/models/g2pModel.js'), () => import('./picovox/lib/models/voiceModel.js'));


// The playback timing (s) - how far ahead of the audio context's time a block starts, and how far ahead of playback
// synthesis runs
const playbackLead = 0.05;
const playbackAhead = 4;


// The noise seed of the text's first sentence (sentence i is rendered with seed + i)
const playbackSeed = 1000;


// Paragraph back goes to the previous paragraph within this many seconds of a paragraph's start (otherwise, to the
// paragraph's start)
const paragraphBackSeconds = 2;


// The voice keys
const voiceKeys = new Set(voices().map(({key}) => key));


// The audio context, created on first playback (from a user gesture, so the browser lets the audio start). It runs at
// the blocks' sample rate, so the blocks join exactly.
let audioContext = null;


// The playback, if any - {key, text, final, voice, rate, sentences, position (the sentence index to play from), playing,
// run (the active synthesis run, or null)}
let playback = null;


// The playback change callback - [callback, options] - called when playback state changes on its own
let playbackChange = null;


// The text's sentences to speak - until the text is final, its last sentence may be incomplete, so it's held back
function playbackSentences(text, final) {
    const sentences = splitSentences(text);
    return final ? sentences : sentences.slice(0, -1);
}


// A sentence's words, for comparing sentences - a sentence's punctuation may change as its text grows (e.g., Markdown
// emphasis spanning sentences is literal until it's closed)
function sentenceWords(sentence) {
    return sentence?.text.replace(/[^\p{L}\p{N}]+/gu, '');
}


// The index of the sentence that's playing (or that's next to play)
function playbackSentence() {
    const {run, position} = playback;
    if (run === null) {
        return position;
    }
    let sentence = run.position;
    for (const [ixSentence, startTime] of run.starts) {
        if (startTime <= audioContext.currentTime) {
            sentence = ixSentence;
        }
    }
    return sentence;
}


// The next task - a promise resolved by a message to the page itself (a timer would be throttled in background tabs)
function nextTask() {
    return new Promise((resolve) => {
        const channel = new MessageChannel();
        channel.port1.onmessage = () => {
            channel.port1.close();
            resolve();
        };
        channel.port2.postMessage(null);
    });
}


// Wait for a run's event - a scheduled block ending, the text changing, or the run stopping
function runWait(run) {
    return new Promise((resolve) => {
        run.waiting.push(resolve);
    });
}


// Wake a run's waiters
function runWake(run) {
    for (const resolve of run.waiting.splice(0)) {
        resolve();
    }
}


// Start a synthesis run from the playback's position
function runStart() {
    const run = {'position': playback.position, 'starts': [], 'sources': new Set(), 'waiting': [], 'stopped': false};
    playback.run = run;
    runPlay(playback, run).catch((error) => {
        // Synthesis failed - stop the playback
        if (!run.stopped) {
            console.error(`Ollama Chat: Voice playback failed: ${error}`);
            runStop();
            playback.playing = false;
            audioContext.suspend();
            playbackOnChange();
        }
    });
}


// Stop the playback's synthesis run, if any
function runStop() {
    const {run} = playback;
    if (run !== null) {
        run.stopped = true;
        for (const source of run.sources) {
            source.onended = null;
            source.stop();
        }
        run.sources.clear();
        runWake(run);
        playback.run = null;
    }
}


// Move the playback to a sentence - playing from there if it's playing, otherwise when it plays
function playbackSeek(position) {
    runStop();
    playback.position = position;
    if (playback.playing) {
        runStart();
    }
}


// The index of a sentence's paragraph's first sentence
function paragraphStart(sentences, ixSentence) {
    let ixStart = ixSentence;
    while (ixStart > 0 && !sentences[ixStart].paragraphStart) {
        ixStart -= 1;
    }
    return ixStart;
}


// Synthesize and schedule the playback's sentences, a block at a time (yielding to the page after each), staying at most
// a few seconds ahead of playback
async function runPlay(pb, run) {
    const {sampleRate} = audioContext;
    const model = await loadModel();
    let time = 0;
    let ixSentence = run.position;
    while (!run.stopped) {
        // Wait for the sentence's text - while its earlier audio plays, or for the first sentence of the run
        while (!run.stopped && ixSentence >= pb.sentences.length && !pb.final && (run.sources.size > 0 || run.starts.length === 0)) {
            await runWait(run);
        }
        if (run.stopped || ixSentence >= pb.sentences.length) {
            break;
        }

        // Synthesize and schedule the sentence's blocks
        const sentence = pb.sentences[ixSentence];
        const pause = sentence.paragraphStart ? paragraphPause : sentencePause;
        const blockOptions = {'voice': pb.voice, 'rate': pb.rate, 'seed': playbackSeed + ixSentence, pause, sampleRate};
        let isFirstBlock = true;
        for (const block of sentenceBlocks(model, speechText(sentence.text), blockOptions)) {
            if (run.stopped) {
                return;
            }
            const buffer = audioContext.createBuffer(1, block.length, sampleRate);
            buffer.copyToChannel(block, 0);
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.onended = () => {
                run.sources.delete(source);
                runWake(run);
            };
            time = Math.max(time, audioContext.currentTime + playbackLead);
            if (isFirstBlock) {
                run.starts.push([ixSentence, time]);
                isFirstBlock = false;
            }
            run.sources.add(source);
            source.start(time);
            time += block.length / sampleRate;
            await nextTask();
            while (!run.stopped && time - audioContext.currentTime > playbackAhead) {
                await runWait(run);
            }
        }
        ixSentence += 1;
    }

    // Wait for the scheduled audio to end
    while (!run.stopped && run.sources.size > 0) {
        await runWait(run);
    }
    if (run.stopped) {
        return;
    }

    // The end of the text - if it's final, the playback ends (back at the start); otherwise, it pauses
    pb.run = null;
    pb.position = (pb.final ? 0 : ixSentence);
    pb.playing = false;
    audioContext.suspend();
    playbackOnChange();
}


// Call the playback change callback, if any, as MarkdownUp calls a runtime event callback
function playbackOnChange() {
    if (playbackChange === null) {
        return;
    }
    const [callback, options] = playbackChange;
    options.runtime.eventHandle(async () => {
        options.statementCount = 0;
        try {
            await callback([], options);
        } catch ({message}) {
            if (options.debug) {
                options.logFn(`Ollama Chat: Error executing voiceSetChange callback: ${message}`);
            }
        }
        options.runtimeUpdateFn();
    });
}


// Validate a voice key
function validateVoice(voice) {
    if (!voiceKeys.has(voice)) {
        throw new ValueArgsError('voice', voice);
    }
}


// $function: voiceClose
// $group: voice
// $doc: Close the voice playback, if any
function voiceClose() {
    if (playback !== null) {
        runStop();
        playback = null;
        audioContext.suspend();
    }
}


// $function: voiceParagraphBack
// $group: voice
// $doc: Move the voice playback back to the start of the paragraph - or, near the paragraph's start, to the start of
// $doc: the previous paragraph
function voiceParagraphBack() {
    if (playback === null || playback.sentences.length === 0) {
        return;
    }
    const {sentences, run} = playback;
    const ixSentence = playbackSentence();

    // Past the end of the text? Back to the start of the last paragraph.
    if (ixSentence >= sentences.length) {
        playbackSeek(paragraphStart(sentences, sentences.length - 1));
        return;
    }

    // Near the start of the paragraph? Back to the start of the previous paragraph.
    const ixStart = paragraphStart(sentences, ixSentence);
    const startTime = run?.starts.find(([ixStarted]) => ixStarted === ixStart)?.[1];
    const isNearStart = ixSentence === ixStart && (startTime === undefined || audioContext.currentTime - startTime < paragraphBackSeconds);
    playbackSeek(isNearStart && ixStart > 0 ? paragraphStart(sentences, ixStart - 1) : ixStart);
}


// $function: voiceParagraphForward
// $group: voice
// $doc: Move the voice playback forward to the start of the next paragraph, if there is one
function voiceParagraphForward() {
    if (playback === null) {
        return;
    }
    const {sentences} = playback;
    const ixSentence = playbackSentence();
    const ixNext = sentences.findIndex((sentence, ix) => ix > ixSentence && sentence.paragraphStart);
    if (ixNext !== -1) {
        playbackSeek(ixNext);
    }
}


// $function: voicePause
// $group: voice
// $doc: Pause the voice playback
function voicePause() {
    if (playback !== null && playback.playing) {
        playback.playing = false;
        audioContext.suspend();
    }
}


// $function: voicePlay
// $group: voice
// $doc: Play (resume) the voice playback
function voicePlay() {
    if (playback !== null && !playback.playing) {
        playback.playing = true;
        audioContext.resume();
        if (playback.run === null) {
            runStart();
        }
    }
}


// $function: voiceRewind
// $group: voice
// $doc: Move the voice playback back to the beginning
function voiceRewind() {
    if (playback !== null) {
        playbackSeek(0);
    }
}


// $function: voiceSetChange
// $group: voice
// $doc: Set the voice playback change callback, called when playback state changes on its own - the end of the text is
// $doc: reached (the playback pauses or ends) or synthesis fails
// $arg callback: The voice playback change callback function
function voiceSetChange(args, options) {
    const [callback] = valueArgsValidate(voiceSetChangeArgs, args);
    playbackChange = [callback, options];
}

const voiceSetChangeArgs = valueArgsModel([
    {'name': 'callback', 'type': 'function'}
]);


// $function: voiceSetRate
// $group: voice
// $doc: Set the voice playback's speaking rate
// $arg rate: The speaking rate, from 0.5 to 2 (1 is normal)
function voiceSetRate(args) {
    const [rate] = valueArgsValidate(voiceSetRateArgs, args);
    if (playback !== null && rate !== playback.rate) {
        playback.rate = rate;
        playbackSeek(playbackSentence());
    }
}

const voiceSetRateArgs = valueArgsModel([
    {'name': 'rate', 'type': 'number', 'gte': 0.5, 'lte': 2}
]);


// $function: voiceSetText
// $group: voice
// $doc: Set the voice playback's text - its text grows as it plays (if the words that have already played change, the
// $doc: playback starts again from the beginning)
// $arg text: The text to read aloud (paragraphs are separated by blank lines)
// $arg final: If true, the text is final (no more text is coming)
function voiceSetText(args) {
    const [text, final] = valueArgsValidate(voiceSetTextArgs, args);
    if (playback === null || (text === playback.text && final === playback.final)) {
        return;
    }

    // Has the text that's already played changed?
    const sentences = playbackSentences(text, final);
    const ixSentence = playbackSentence();
    let changed = false;
    for (let ix = 0; ix < ixSentence; ix++) {
        if (sentenceWords(sentences[ix]) !== sentenceWords(playback.sentences[ix])) {
            changed = true;
            break;
        }
    }

    // Update the text - if the text that's already played changed, start again from the beginning
    playback.text = text;
    playback.final = final;
    playback.sentences = sentences;
    if (changed) {
        runStop();
        playback.position = 0;
        if (playback.playing) {
            runStart();
        }
    } else if (playback.run !== null) {
        runWake(playback.run);
    }
}

const voiceSetTextArgs = valueArgsModel([
    {'name': 'text', 'type': 'string'},
    {'name': 'final', 'type': 'boolean'}
]);


// $function: voiceSetVoice
// $group: voice
// $doc: Set the voice playback's voice
// $arg voice: The voice key
function voiceSetVoice(args) {
    const [voice] = valueArgsValidate(voiceSetVoiceArgs, args);
    validateVoice(voice);
    if (playback !== null && voice !== playback.voice) {
        playback.voice = voice;
        playbackSeek(playbackSentence());
    }
}

const voiceSetVoiceArgs = valueArgsModel([
    {'name': 'voice', 'type': 'string'}
]);


// $function: voiceSpeak
// $group: voice
// $doc: Start reading text aloud (from a user gesture, so the browser lets the audio start), closing any playback
// $arg key: The playback's key value - returned by voiceState
// $arg text: The text to read aloud (paragraphs are separated by blank lines)
// $arg final: If true, the text is final (no more text is coming)
// $arg voice: The voice key
// $arg rate: The speaking rate, from 0.5 to 2 (1 is normal)
function voiceSpeak(args) {
    const [key, text, final, voice, rate] = valueArgsValidate(voiceSpeakArgs, args);
    validateVoice(voice);
    voiceClose();
    if (audioContext === null) {
        audioContext = new AudioContext({'sampleRate': defaultSampleRate});
    }
    audioContext.resume();
    playback = {key, text, final, voice, rate, 'sentences': playbackSentences(text, final), 'position': 0, 'playing': true, 'run': null};
    runStart();
}

const voiceSpeakArgs = valueArgsModel([
    {'name': 'key'},
    {'name': 'text', 'type': 'string'},
    {'name': 'final', 'type': 'boolean'},
    {'name': 'voice', 'type': 'string'},
    {'name': 'rate', 'type': 'number', 'gte': 0.5, 'lte': 2}
]);


// $function: voiceState
// $group: voice
// $doc: Get the voice playback state
// $return: The voice playback state object (key, playing, voice, and rate), or null if there's no playback
function voiceState() {
    if (playback === null) {
        return null;
    }
    return {'key': playback.key, 'playing': playback.playing, 'voice': playback.voice, 'rate': playback.rate};
}


// $function: voiceVoices
// $group: voice
// $doc: Get the voices
// $return: The array of voice objects (key, label, sex, and description) - the default voice first
function voiceVoices() {
    return voices();
}


// The voice playback BareScript runtime functions (MarkdownUp globals)
export const voiceFunctions = {
    voiceClose,
    voiceParagraphBack,
    voiceParagraphForward,
    voicePause,
    voicePlay,
    voiceRewind,
    voiceSetChange,
    voiceSetRate,
    voiceSetText,
    voiceSetVoice,
    voiceSpeak,
    voiceState,
    voiceVoices
};
