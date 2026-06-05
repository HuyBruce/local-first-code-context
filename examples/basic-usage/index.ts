import {
    Context,
    MilvusVectorDatabase,
    MilvusRestfulVectorDatabase,
    AstCodeSplitter,
    LangChainCodeSplitter
} from '@zilliz/claude-context-core';
import { envManager } from '@zilliz/claude-context-core';
import * as path from 'path';

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

        console.log(`\nStarting to index: ${codebasePath}`);

        const hasExistingIndex = await context.hasIndex(codebasePath);
        if (hasExistingIndex) {
            console.log('Existing index found, clearing it first...');
            await context.clearIndex(codebasePath);
        }

        const indexStats = await context.indexCodebase(codebasePath);
        console.log(`\nIndexing stats: ${indexStats.indexedFiles} files, ${indexStats.totalChunks} code chunks`);

        console.log('\nPerforming semantic search...');

        const queries = [
            'vector database operations',
            'code splitting functions',
            'embedding generation',
            'environment variable configuration'
        ];

        for (const query of queries) {
            console.log(`\nSearch: "${query}"`);
            const results = await context.semanticSearch(codebasePath, query, 3, 0.3);

            if (results.length === 0) {
                console.log('   No relevant results found');
                continue;
            }

            results.forEach((result, index) => {
                console.log(`   ${index + 1}. Similarity: ${(result.score * 100).toFixed(2)}%`);
                console.log(`      File: ${path.join(codebasePath, result.relativePath)}`);
                console.log(`      Language: ${result.language}`);
                console.log(`      Lines: ${result.startLine}-${result.endLine}`);
                console.log(`      Preview: ${result.content.substring(0, 100)}...`);
            });
        }

        console.log('\nExample completed successfully!');
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
            console.log('   - DEMO_CODEBASE_PATH: Optional path to index');
            console.log('   - EMBEDDING_BATCH_SIZE: Batch size for embeddings');
        }

        process.exit(1);
    }
}

if (require.main === module) {
    main().catch(console.error);
}

export { main };
