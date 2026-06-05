import {
    Context,
    MilvusVectorDatabase,
    MilvusRestfulVectorDatabase,
    AstCodeSplitter,
    LangChainCodeSplitter
} from '@zilliz/claude-context-core';
import { envManager } from '@zilliz/claude-context-core';
import { SemanticSearchResult } from '@zilliz/claude-context-core';
import * as path from 'path';
import * as readline from 'readline/promises';

try {
    require('dotenv').config();
} catch (error) {
    // dotenv is optional for this demo.
}

async function main() {
    console.log('Claude Context Local Demo');
    console.log('=========================');

    try {
        process.env.EMBEDDING_BATCH_SIZE = process.env.EMBEDDING_BATCH_SIZE || '10';
        process.env.HYBRID_MODE = process.env.HYBRID_MODE || 'false';

        const useRestfulApi = (process.env.MILVUS_USE_RESTFUL || '').toLowerCase() === 'true';
        const milvusAddress = envManager.get('MILVUS_ADDRESS') || 'localhost:19530';
        const milvusToken = envManager.get('MILVUS_TOKEN');
        const splitterType = envManager.get('SPLITTER_TYPE')?.toLowerCase() || 'ast';

        console.log(`Using ${useRestfulApi ? 'RESTful API' : 'gRPC'} implementation`);
        console.log(`Connecting to Milvus at: ${milvusAddress}`);

        const vectorDatabase = useRestfulApi
            ? new MilvusRestfulVectorDatabase({
                address: milvusAddress,
                ...(milvusToken && { token: milvusToken })
            })
            : new MilvusVectorDatabase({
                address: milvusAddress,
                ...(milvusToken && { token: milvusToken })
            });

        const codeSplitter = splitterType === 'langchain'
            ? new LangChainCodeSplitter(1000, 200)
            : new AstCodeSplitter(2500, 300);

        const context = new Context({
            vectorDatabase,
            codeSplitter,
            supportedExtensions: ['.ts', '.js', '.py', '.java', '.cpp', '.go', '.rs']
        });

        const codebasePath = process.env.DEMO_CODEBASE_PATH
            ? path.resolve(process.env.DEMO_CODEBASE_PATH)
            : path.join(__dirname, '../../packages/core/src');

        console.log(`\nPreparing index: ${codebasePath}`);

        const hasExistingIndex = await context.hasIndex(codebasePath);
        const shouldReindex = (process.env.DEMO_REINDEX || '').toLowerCase() === 'true';
        if (hasExistingIndex && shouldReindex) {
            console.log('Existing index found, clearing it because DEMO_REINDEX=true...');
            await context.clearIndex(codebasePath);
        }

        if (!hasExistingIndex || shouldReindex) {
            const indexStats = await context.indexCodebase(codebasePath);
            console.log(`\nIndexing stats: ${indexStats.indexedFiles} files, ${indexStats.totalChunks} code chunks`);
        } else {
            console.log('Existing index found. Set DEMO_REINDEX=true to rebuild it.');
        }

        const chatModel = process.env.CHAT_MODEL || 'llama3.2:3b';
        console.log(`\nLocal code chat mode using Ollama model: ${chatModel}`);
        console.log('Ask about this codebase, or type "exit" to quit.');
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        while (true) {
            let query: string;
            try {
                query = (await rl.question('\nAsk> ')).trim();
            } catch (error) {
                break;
            }

            if (!query || query.toLowerCase() === 'exit' || query.toLowerCase() === 'quit') {
                break;
            }

            await runChat(context, codebasePath, query, chatModel);
        }

        rl.close();
        console.log('\nDemo finished.');
    } catch (error) {
        console.error('Error occurred:', error);

        if (error instanceof Error) {
            if (error.message.includes('API key')) {
                console.log('\nCheck embedding provider configuration.');
                console.log('   Local example: EMBEDDING_PROVIDER=Ollama and OLLAMA_MODEL=nomic-embed-text');
            } else if (error.message.includes('Milvus') || error.message.includes('connect')) {
                console.log('\nCheck that Milvus is running at localhost:19530.');
            }

            console.log('\nEnvironment variables:');
            console.log('   - EMBEDDING_PROVIDER: Ollama');
            console.log('   - OLLAMA_MODEL: nomic-embed-text');
            console.log('   - OLLAMA_HOST: http://127.0.0.1:11434');
            console.log('   - MILVUS_ADDRESS: localhost:19530');
            console.log('   - CHAT_MODEL: Ollama chat model (default: llama3.2:3b)');
            console.log('   - DEMO_CODEBASE_PATH: Optional path to index');
            console.log('   - DEMO_REINDEX: Set true to rebuild the Milvus index');
            console.log('   - EMBEDDING_BATCH_SIZE: Batch size for embeddings');
        }

        process.exit(1);
    }
}

async function runSearch(context: Context, codebasePath: string, query: string): Promise<void> {
    console.log(`\nSearch: "${query}"`);
    const results = await context.semanticSearch(codebasePath, query, 3, 0.3);

    if (results.length === 0) {
        console.log('   No relevant results found');
        return;
    }

    results.forEach((result, index) => {
        console.log(`   ${index + 1}. Similarity: ${(result.score * 100).toFixed(2)}%`);
        console.log(`      File: ${path.join(codebasePath, result.relativePath)}`);
        console.log(`      Language: ${result.language}`);
        console.log(`      Lines: ${result.startLine}-${result.endLine}`);
        console.log(`      Preview: ${result.content.substring(0, 100)}...`);
    });
}

async function runChat(
    context: Context,
    codebasePath: string,
    query: string,
    chatModel: string
): Promise<void> {
    console.log(`\nRetrieving code context for: "${query}"`);
    const results = await context.semanticSearch(codebasePath, query, 5, 0.25);

    if (results.length === 0) {
        console.log('No relevant local code context found.');
        return;
    }

    const prompt = buildPrompt(query, codebasePath, results);
    const answer = await generateWithOllama(chatModel, prompt);

    console.log('\nAnswer');
    console.log(answer.trim());

    console.log('\nSources');
    results.slice(0, 5).forEach((result, index) => {
        console.log(
            `${index + 1}. ${path.join(codebasePath, result.relativePath)}:${result.startLine}-${result.endLine} ` +
            `(similarity ${(result.score * 100).toFixed(2)}%)`
        );
    });
}

function buildPrompt(query: string, codebasePath: string, results: SemanticSearchResult[]): string {
    const contextBlocks = results.slice(0, 5).map((result, index) => {
        const sourcePath = path.join(codebasePath, result.relativePath);
        const content = result.content.slice(0, 1400);
        return [
            `Source ${index + 1}: ${sourcePath}:${result.startLine}-${result.endLine}`,
            '```',
            content,
            '```'
        ].join('\n');
    }).join('\n\n');

    return [
        'You are a local codebase assistant.',
        'Answer the user question using only the provided code context.',
        'In this project, "code splitting" means splitting source files into searchable chunks for indexing, not frontend bundle splitting.',
        'If the context is insufficient, say what is missing.',
        'Be concise and cite source numbers like [Source 1].',
        '',
        `Question: ${query}`,
        '',
        'Code context:',
        contextBlocks,
        '',
        'Answer:'
    ].join('\n');
}

async function generateWithOllama(model: string, prompt: string): Promise<string> {
    const host = (envManager.get('OLLAMA_HOST') || 'http://127.0.0.1:11434').replace(/\/$/, '');
    const response = await fetch(`${host}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model,
            prompt,
            stream: false,
            options: {
                temperature: 0.2,
                top_p: 0.9
            }
        })
    });

    if (!response.ok) {
        const body = await response.text();
        throw new Error(`Ollama generation failed (${response.status}): ${body}`);
    }

    const data = await response.json() as { response?: string };
    return data.response || '';
}

if (require.main === module) {
    main().catch(console.error);
}

export { main };
