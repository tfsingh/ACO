import { createClient } from 'redis';

const client = createClient({
    password: process.env.REDIS_PASSWORD,
    socket: {
        host: process.env.REDIS_ENDPOINT,
        //@ts-ignore
        port: process.env.REDIS_PORT
    }
});

client.on('error', (err) => console.log(err));

if (!client.isOpen) {
    client.connect()
}

const whitelist_key = process.env.WHITELIST_KEY;

export async function addToWhitelist(email: string) {
    if (whitelist_key) {
        await client.SADD(whitelist_key, email);
    } else {
        console.error('WHITELIST_KEY is not defined');
    }
}